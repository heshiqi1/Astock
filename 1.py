#!/usr/bin/env python3
"""
通达信选股策略回测 - 严格长下影线优化版

完整选股 / 入出场说明见同目录 STRATEGY.md（请与该文件同步维护）。

对照回测：A/B/C 三组；长下影 A=确认K收盘，B=信号日收盘，C=次日开盘突破信号收盘+offset；
其它信号仅 A/B 有效（C 不适用）。报告与 Excel 对比胜率。

选股：任意信号须 **同时** 满足「多头发散 + RSI」与对应形态（长下影线或阳包阴）；长下影线形态本身只做 K 线 1～4 步，隔日确认仅用于策略 A 回测。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import os
import sys
import zipfile
import struct
import warnings
import multiprocessing

warnings.filterwarnings('ignore')


def _ensure_utf8_stdio():
    """Windows 默认 GBK 控制台打印 emoji 会报错，尽量切到 UTF-8。"""
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    MPF_AVAILABLE = True
except ImportError:
    MPF_AVAILABLE = False


# ==================== 配置 ====================
CONFIG = {
    'zip_file': 'tdx_data/tdx_day_data.zip',
    'charts_dir': 'trade_charts',
    # 导出 K 线总张数上限（同时导出 A+B+C 时约为 3×有效成交，可适当调大）
    'max_charts': 4000,
    'chart_bars': 100,
    
    'target_date': None,
    'start_date': '20200101',  # 【修复】改为字符串
    'end_date': '20201229',    # 【修复】改为字符串
    
    'hold_days_max': 60,
    'take_profit_pct': 0.15,
    'stop_loss_pct': 0.03,
    'use_trailing_stop': True,
    'trailing_stop_pct': 0.10,
    # True：先曾收盘在 exit_below_ma_period 均线之上，之后收盘跌破该线则按收盘卖出（列 MA{n}）
    'exit_on_close_below_ma20': True,
    
    'debug_mode': False,
    'debug_max_detail': 10,
    # 可选：与脚本同目录的 JSON（键为 6 位代码、值为中文简称），用于图片文件名
    'stock_names_json': 'stock_names.json',
    # simulate() 默认入场方式（主流程会同时跑对照组，此键仅给单独调用 simulate 时用）
    'entry_mode_default': 'strategy_a',
    # 选股：4 条均线周期（须递增），对应 close > ma[0] > ma[1] > ma[2] > ma[3]
    'ma_periods': [5, 10, 20, 30],
    'rsi_period': 14,
    # 当日 RSI 须小于该值才允许多头信号（超买过滤）
    'rsi_max_for_signal': 75,
    # 出场「跌破均线」使用的周期，列名为 MA{n}（会随 ma_periods 一并计算）
    'exit_below_ma_period': 20,
    # 策略C：长下影次日开盘须高于「信号日收盘价 + 该偏移」才按开盘价入场（与 signal_high 突破区分）
    'breakout_above_signal_close_offset': 0.05,
}


def _ma_col(period: int) -> str:
    return f"MA{int(period)}"


def _min_df_bars_for_indicators() -> int:
    """技术面所需最小时长：不低于 35 以兼容原逻辑。"""
    periods = CONFIG.get('ma_periods', [5, 10, 20, 30])
    exit_p = int(CONFIG.get('exit_below_ma_period', 20))
    rsi_p = int(CONFIG.get('rsi_period', 14))
    need = max(max(periods), exit_p, rsi_p) + 5
    return max(35, need)


# 股票代码 -> 简称（由 load_stock_names 填充）
STOCK_NAMES: Dict[str, str] = {}


def load_stock_names() -> None:
    """从 stock_names.json 加载股票简称（可选文件，不存在则文件名用 6 位代码）。"""
    global STOCK_NAMES
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CONFIG.get('stock_names_json', 'stock_names.json'),
    )
    if not os.path.isfile(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            key = str(k).strip()
            if len(key) >= 6 and key[:6].isdigit():
                STOCK_NAMES[key[:6]] = str(v).strip()
    except Exception:
        pass


def _format_date_for_filename(yyyymmdd: str) -> str:
    """20260309 -> 2026-03-09"""
    s = str(yyyymmdd).replace('-', '')
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return str(yyyymmdd)


def _sanitize_filename_part(s: str, max_len: int = 48) -> str:
    """Windows 文件名非法字符替换为下划线。"""
    bad = '\\/:*?"<>|\n\r\t'
    out = ''.join('_' if c in bad else c for c in str(s).strip())
    out = out.replace(' ', '_')
    if len(out) > max_len:
        out = out[:max_len]
    return out or 'x'


def _chart_filename(trade: 'Trade', signal: Dict) -> str:
    """入场方式-股票名称-日期-信号类型-盈亏.png（入场方式区分 A/B，避免覆盖）"""
    name = signal.get('name') or trade.name or trade.code
    name = STOCK_NAMES.get(trade.code, name)
    dt = _format_date_for_filename(trade.signal_date)
    stype = trade.signal_type or '信号'
    pnl_part = f"{trade.pnl_pct * 100:+.2f}%"
    es = getattr(trade, 'entry_style', '') or ''
    parts = []
    if es:
        parts.append(_sanitize_filename_part(es, 20))
    parts.extend([
        _sanitize_filename_part(name, 32),
        _sanitize_filename_part(dt, 12),
        _sanitize_filename_part(stype, 24),
        _sanitize_filename_part(pnl_part, 16),
    ])
    return '-'.join(parts) + '.png'


# ==================== 代码过滤 ====================
def filter_stock_code(code: str) -> bool:
    if not code or len(code) != 6 or not code.isdigit():
        return False
    if code.startswith(('600', '601', '603', '605')):
        return True
    if code.startswith(('000', '001', '002', '003')):
        return True
    if code.startswith(('300', '301')):
        return True
    return False


@dataclass
class Trade:
    code: str
    name: str
    signal_date: str
    entry_date: str
    exit_date: str
    signal_price: float
    entry_price: float
    exit_price: float
    signal_type: str
    exit_reason: str
    pnl: float
    pnl_pct: float
    hold_days: int
    entry_style: str = ""  # 确认K收盘买入 / 长下影当日收盘买入


# ==================== 数据读取 ====================
def parse_day_binary(data: bytes) -> Optional[pd.DataFrame]:
    """【修复】正确解析.day二进制文件"""
    if len(data) < 32:
        return None
    
    record_size = 32
    num_records = len(data) // record_size
    
    dates = np.empty(num_records, dtype='datetime64[ns]')
    opens = np.empty(num_records, dtype=np.float32)
    highs = np.empty(num_records, dtype=np.float32)
    lows = np.empty(num_records, dtype=np.float32)
    closes = np.empty(num_records, dtype=np.float32)
    volumes = np.empty(num_records, dtype=np.int64)
    
    for i in range(num_records):
        offset = i * record_size
        # 【关键修复】正确解包并索引
        record = struct.unpack_from('IIIIIIII', data, offset)
        # 通达信 .day: date, open*100, high*100, low*100, close*100, amount, volume, reserved
        date_int = record[0]
        open_raw = record[1]
        high_raw = record[2]
        low_raw = record[3]
        close_raw = record[4]
        vol = record[6]

        dates[i] = np.datetime64(
            f"{date_int // 10000:04d}-{(date_int % 10000) // 100:02d}-{date_int % 100:02d}"
        )
        opens[i] = open_raw / 100.0
        highs[i] = high_raw / 100.0
        lows[i] = low_raw / 100.0
        closes[i] = close_raw / 100.0
        volumes[i] = vol

    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    ).sort_values("date").reset_index(drop=True)


_zip_cache = {}

def read_stock_from_zip(zip_path: str, file_path: str) -> Optional[pd.DataFrame]:
    try:
        if zip_path not in _zip_cache:
            _zip_cache[zip_path] = zipfile.ZipFile(zip_path, 'r')
        zf = _zip_cache[zip_path]
        
        if file_path.endswith('.day'):
            with zf.open(file_path) as f:
                return parse_day_binary(f.read())
        elif file_path.endswith('.csv'):
            with zf.open(file_path) as f:
                df = pd.read_csv(f, encoding='gbk')
                col_map = {'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}
                df = df.rename(columns=col_map)
                df['date'] = pd.to_datetime(df['date'])
                return df
    except Exception as e:
        if CONFIG['debug_mode']:
            print(f"    [读取错误] {e}")
    return None


# ==================== 技术指标 ====================
def calculate_ma(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close'].astype(float).values
    periods_cfg = CONFIG.get('ma_periods', [5, 10, 20, 30])
    exit_p = int(CONFIG.get('exit_below_ma_period', 20))
    all_periods = sorted({int(p) for p in list(periods_cfg) + [exit_p]})
    for p in all_periods:
        df[_ma_col(p)] = pd.Series(close).rolling(p, min_periods=1).mean().values
    return df


def calculate_rsi(prices: np.ndarray, period: Optional[int] = None) -> np.ndarray:
    """RSI；period 默认取 CONFIG['rsi_period']。"""
    if period is None:
        period = int(CONFIG.get('rsi_period', 14))
    if len(prices) < 2:
        return np.full(len(prices), 50.0, dtype=float)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gains = pd.Series(gains).rolling(window=period, min_periods=1).mean().values
    avg_losses = pd.Series(losses).rolling(window=period, min_periods=1).mean().values
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[30], rsi])


# ==================== 【核心】长下影线形态 + 隔日确认（拆分）====================
def check_long_shadow_shape(df: pd.DataFrame, idx: int, show_detail: bool = False) -> bool:
    """
    长下影线（锤子线）仅 K 线形态 1～4，不含均线/RSI、不含隔日确认K。
    实际选股在 check_signal_advanced 中须与本函数 True **且** bullish_trend 同日成立。
    1. 回调：low < min(前3根low)
    2. 不大涨：(close-open)/open < 5%
    3. 形态：下影线 >= 实体*0.8，上影线 <= 实体*0.2，收阳
    4. 重叠：当前实体与前实体重叠 < 30%
    """
    if idx < 4 or idx >= len(df):
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    # 1. 回调位置：当前low低于前3根low的最低点
    prev3_lows = df.iloc[idx-3:idx]['low'].values
    if curr['low'] >= prev3_lows.min() * 0.999:
        if show_detail:
            print(f"     ❌ 非回调: 当前low{curr['low']:.2f} >= 前3低{prev3_lows.min():.2f}")
        return False
    
    # 2. 不能大涨 (<5%)
    _open = float(curr['open'])
    if abs(_open) < 1e-9:
        return False
    change_pct = (curr['close'] - curr['open']) / _open
    if change_pct > 0.05:
        if show_detail:
            print(f"     ❌ 大涨: {change_pct*100:.1f}% > 5%")
        return False
    
    # 3. 形态计算
    body = abs(curr['close'] - curr['open'])
    upper_shadow = curr['high'] - max(curr['close'], curr['open'])
    lower_shadow = min(curr['close'], curr['open']) - curr['low']
    
    if body < 0.01:  # 避免除零
        if show_detail:
            print(f"     ❌ 实体太小")
        return False
    
    # 【关键修复】长下影线：下影线长（>=实体0.8倍），上影线短（<=实体0.2倍）
    if lower_shadow < body * 0.8:  # 下影线要占主体
        if show_detail:
            print(f"     ❌ 下影线不够长: {lower_shadow:.2f} < {body*0.8:.2f} (实体{body:.2f})")
        return False
    
    if upper_shadow > body * 0.2:  # 上影线要短
        if show_detail:
            print(f"     ❌ 上影线太长: {upper_shadow:.2f} > {body*0.2:.2f}")
        return False
    
    # 收阳
    if curr['close'] <= curr['open']:
        if show_detail:
            print(f"     ❌ 非阳线")
        return False
    
    # 4. 重叠检查
    curr_body_top = max(curr['open'], curr['close'])
    curr_body_bottom = min(curr['open'], curr['close'])
    prev_body_top = max(prev['open'], prev['close'])
    prev_body_bottom = min(prev['open'], prev['close'])
    
    overlap_top = min(curr_body_top, prev_body_top)
    overlap_bottom = max(curr_body_bottom, prev_body_bottom)
    overlap = max(0, overlap_top - overlap_bottom)
    
    if body > 0 and overlap / body > 0.3:
        if show_detail:
            print(f"     ❌ 重叠太多: {overlap/body*100:.0f}% > 30%")
        return False
    
    if show_detail:
        print(f"     ✔️ 长下影形态: 下影{lower_shadow:.2f}>=实体{body:.2f}, 上影{upper_shadow:.2f}<=0.2实体")
    return True


def check_signal_k_confirmation(df: pd.DataFrame, idx: int, show_detail: bool = False) -> bool:
    """
    隔日信号K确认（原规则第5步）：idx 为长下影线日，检验 idx+1 为强阳线、非十字星、实体 >= 信号日实体*0.9。
    策略A：仅当本函数为 True 时，于 idx+1 日收盘价买入。
    """
    if idx < 4 or idx + 1 >= len(df):
        return False
    curr = df.iloc[idx]
    next_c = df.iloc[idx + 1]
    body = abs(curr['close'] - curr['open'])
    if body < 0.01:
        return False
    
    next_body = abs(next_c['close'] - next_c['open'])
    next_upper = next_c['high'] - max(next_c['close'], next_c['open'])
    next_lower = min(next_c['close'], next_c['open']) - next_c['low']
    next_shadow = next_upper + next_lower
    
    if next_body <= next_shadow * 0.8:
        if show_detail:
            print(f"     ❌ 确认K线太弱: 实体{next_body:.2f} <= 影线{next_shadow:.2f}")
        return False
    
    if next_c['close'] <= next_c['open']:
        if show_detail:
            print(f"     ❌ 确认K线非阳线")
        return False
    
    if next_body < body * 0.9:
        if show_detail:
            print(f"     ❌ 确认K线实体不够大: {next_body:.2f} < {body*0.9:.2f}")
        return False
    
    if show_detail:
        print(f"     ✔️ 隔日确认K: 实体{next_body:.2f} >= 信号日实体0.9倍")
    return True


# ==================== 信号检测 ====================
def check_signal_advanced(df: pd.DataFrame, valid_indices: np.ndarray, code: str, show_detail: bool = False) -> List[Tuple[int, str]]:
    """
    信号检测：任一类信号必须 **同时满足**
    （1）当日 bullish_trend：收盘在 CONFIG['ma_periods'] 四条均线上方且 RSI < CONFIG['rsi_max_for_signal']；
    （2）对应形态：长下影线形态 或 阳包阴（后者代码分支见内联）。
    """
    signals = []
    
    min_bars = _min_df_bars_for_indicators()
    if len(df) < min_bars:
        return signals
    
    df = df.reset_index(drop=True)
    close = df['close'].values
    periods = CONFIG.get('ma_periods', [5, 10, 20, 30])
    if len(periods) != 4:
        raise ValueError("CONFIG['ma_periods'] 必须为 4 个整数均线周期（递增）")
    mcols = [_ma_col(p) for p in periods]
    for c in mcols:
        if c not in df.columns:
            return signals
    m1, m2, m3, m4 = (df[c].values for c in mcols)
    rsi = calculate_rsi(close)
    rsi_cap = float(CONFIG.get('rsi_max_for_signal', 75))
    
    # 选股硬条件：多头发散 + RSI（与形态条件同时满足才算信号）
    bullish_trend = (close > m1) & (m1 > m2) & (m2 > m3) & (m3 > m4) & (rsi < rsi_cap)
    
    first_detail_shown = False
    
    for idx in valid_indices:
        idx = int(idx)
        if idx < 5 or idx >= len(df):
            continue
        
        if show_detail and not first_detail_shown and idx >= 30:
            row = df.iloc[idx]
            print(f"\n  🔍 {code} 检查点 (索引{idx}, {row['date']}):")
            print(f"     价格: O{row['open']:.2f} H{row['high']:.2f} L{row['low']:.2f} C{row['close']:.2f}")
            first_detail_shown = True
        
        # 策略1：长下影线 = 形态(check_long_shadow_shape) ∧ 多头发散+RSI(bullish_trend)；隔日在策略A回测里验证
        is_long_shadow = check_long_shadow_shape(df, idx, show_detail and not first_detail_shown)
        if is_long_shadow and bullish_trend[idx]:
            signals.append((idx, '长下影线'))
            if show_detail:
                print(f"     ✅ 长下影线信号确认 @ {df.iloc[idx]['date']}")
            continue
        
        # 策略2：阳包阴（简化）
        if idx >= 4 and bullish_trend[idx]:
            try:
                d = df.iloc[idx-4:idx+1]
                if (d.iloc['open'] > d.iloc['close'] and  # 前1天阴线
                    d.iloc['close'] > d.iloc['open'] and  # 包络
                    d.iloc['close'] > d.iloc['open']):    # 当天阳线
                    signals.append((idx, '阳包阴'))
            except:
                pass
    
    return signals


# ==================== 筛选 ====================
def screen_stocks(zip_path: str, stocks: List[Tuple[str, str]], start_date: str, end_date: str) -> List[Dict]:
    signals = []
    total = len(stocks)
    
    print(f"\n{'='*80}")
    print(f"🔍 开始筛选（长下影线策略）")
    print(f"   股票总数: {total}")
    print(f"   回测区间: {start_date} ~ {end_date}")
    print(f"{'='*80}")
    
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    
    processed = 0
    
    for i, (code, file_path) in enumerate(stocks):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"   进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) | 已发现信号: {len(signals)}")
        
        show_detail = CONFIG['debug_mode'] and processed < CONFIG['debug_max_detail']
        
        df = read_stock_from_zip(zip_path, file_path)
        if df is None or len(df) < _min_df_bars_for_indicators():
            continue
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        date_min, date_max = df['date'].min(), df['date'].max()
        if date_max < start_dt or date_min > end_dt:
            continue
        
        df = calculate_ma(df)
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        if not mask.any():
            continue
        
        valid_indices = df[mask].index.values
        
        if show_detail:
            print(f"\n  [{processed+1}] 分析 {code} | 数据{date_min.strftime('%Y%m%d')}~{date_max.strftime('%Y%m%d')}")
            processed += 1
        
        signal_list = check_signal_advanced(df, valid_indices, code, show_detail)
        
        for idx, signal_type in signal_list:
            row = df.iloc[idx]
            pressure = df.iloc[max(0, idx-20):idx]['high'].max() if idx > 0 else row['high']
            
            signals.append({
                'code': code,
                'name': STOCK_NAMES.get(code, code),
                'signal_date': row['date'].strftime('%Y%m%d'),
                'signal_type': signal_type,
                'signal_price': row['close'],
                'signal_high': row['high'],
                'signal_low': row['low'],
                'pressure': pressure,
                'file_path': file_path,
                'signal_idx': idx
            })
            print(f"   ✅ {code} | {row['date'].strftime('%Y%m%d')} | {signal_type}")
    
    if zip_path in _zip_cache:
        _zip_cache[zip_path].close()
        del _zip_cache[zip_path]
    
    print(f"\n{'='*80}")
    print(f"📊 筛选完成: 共发现 {len(signals)} 个信号")
    print(f"{'='*80}")
    
    return signals


# ==================== 交易模拟（简化）====================
class TradeSimulator:
    """交易模拟：A/B/C 对照。
    长下影线：A=确认K收盘；B=信号日收盘；C=次日开盘突破信号日收盘+offset 则开盘入。
    其它信号类型：A=次日开盘突破前高+0.05；B=信号日收盘；C=不适用（记未成交说明）。
    """
    ENTRY_STRATEGY_A = "strategy_a"  # A（长下影走确认K收盘，否则走次日开盘突破前高）
    ENTRY_SIGNAL_CLOSE = "signal_close"  # B：信号日收盘
    ENTRY_STRATEGY_C = "strategy_c"  # C：长下影次日开盘突破收盘+offset（仅长下影线）

    def __init__(self, charts_dir: str = None):
        self.charts_dir = charts_dir or CONFIG['charts_dir']
        self.plotted_count = 0
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)

    def _empty_trade(self, signal: Dict, code: str, nm: str, reason: str, entry_style: str) -> Trade:
        return Trade(
            code, nm, signal['signal_date'], '', '',
            signal['signal_price'], 0, 0, signal['signal_type'],
            reason, 0, 0, 0, entry_style,
        )

    def simulate(self, signal: Dict, df: pd.DataFrame, entry_mode: str = None) -> Trade:
        """entry_mode: 'strategy_a' | 'signal_close' | 'strategy_c'；旧键 confirm_close/next_day_breakout 映射到 A。"""
        if entry_mode is None:
            entry_mode = CONFIG.get("entry_mode_default", self.ENTRY_STRATEGY_A)
        if entry_mode in ("confirm_close", "next_day_breakout"):
            entry_mode = self.ENTRY_STRATEGY_A  # 兼容旧配置
        idx = signal['signal_idx']
        code = signal['code']
        nm = signal.get('name', code)
        if entry_mode == self.ENTRY_SIGNAL_CLOSE:
            return self._simulate_signal_close(signal, df, idx, code, nm)
        if entry_mode == self.ENTRY_STRATEGY_C:
            return self._simulate_long_shadow_open_break_signal_close(signal, df, idx, code, nm)
        # A 组：长下影线 → 确认K日收盘；否则 → 原次日开盘突破
        if signal.get("signal_type") == "长下影线":
            return self._simulate_confirm_close(signal, df, idx, code, nm)
        return self._simulate_next_day_breakout(signal, df, idx, code, nm)

    def _exit_loop(
        self,
        df: pd.DataFrame,
        signal: Dict,
        entry_price: float,
        entry_date: str,
        entry_idx: int,
        loop_start_idx: int,
        max_idx: int,
        entry_style: str,
    ) -> Trade:
        """从 loop_start_idx 起逐日检查止盈止损（entry_idx 为信号K索引，用于 hold_days）"""
        stop_loss = entry_price * (1 - CONFIG['stop_loss_pct'])
        take_profit = entry_price * (1 + CONFIG['take_profit_pct'])
        highest_price = entry_price
        pressure = signal['pressure']
        # 跌破 exit_below_ma_period 均线：仅当曾收在该线之上，之后再跌破才触发
        was_above_ma20 = False

        for i in range(loop_start_idx, max_idx):
            day = df.iloc[i]
            hold_days = i - entry_idx

            if day['high'] > highest_price:
                highest_price = day['high']

            if CONFIG['use_trailing_stop']:
                profit_pct = (highest_price - entry_price) / entry_price
                if profit_pct >= 0.10:
                    trailing_stop = highest_price * 0.95
                    if day['low'] <= trailing_stop:
                        return self._create_trade(
                            signal, entry_price, entry_date, day, hold_days,
                            trailing_stop, '移动止盈', entry_style,
                        )

            if day['open'] <= stop_loss:
                return self._create_trade(
                    signal, entry_price, entry_date, day, hold_days,
                    day['open'], '止损跳空', entry_style,
                )
            if day['low'] <= stop_loss:
                return self._create_trade(
                    signal, entry_price, entry_date, day, hold_days,
                    stop_loss, '止损', entry_style,
                )

            if day['high'] >= take_profit:
                exit_price = max(day['open'], take_profit)
                return self._create_trade(
                    signal, entry_price, entry_date, day, hold_days,
                    exit_price, '止盈-15%', entry_style,
                )

            if day['high'] >= pressure and day['high'] > entry_price * 1.03:
                exit_price = max(day['open'], pressure)
                return self._create_trade(
                    signal, entry_price, entry_date, day, hold_days,
                    exit_price, '止盈-压力位', entry_style,
                )

            # 收盘价跌破指定周期均线则离场：须先曾收在该均线之上，再跌破才卖（排在止盈/止损之后）
            exit_ma_col = _ma_col(int(CONFIG.get('exit_below_ma_period', 20)))
            if CONFIG.get('exit_on_close_below_ma20', True) and exit_ma_col in df.columns:
                ma_exit = float(day[exit_ma_col])
                if not np.isnan(ma_exit):
                    c = float(day['close'])
                    if c >= ma_exit:
                        was_above_ma20 = True
                    elif was_above_ma20 and c < ma_exit:
                        return self._create_trade(
                            signal, entry_price, entry_date, day, hold_days,
                            c, f'跌破{exit_ma_col}', entry_style,
                        )

        last_day = df.iloc[max_idx - 1]
        hold_days = max_idx - 1 - entry_idx
        return self._create_trade(
            signal, entry_price, entry_date, last_day, hold_days,
            last_day['close'], '到期平仓', entry_style,
        )

    def _simulate_confirm_close(self, signal: Dict, df: pd.DataFrame, idx: int, code: str, nm: str) -> Trade:
        """长下影线策略A：隔日须满足确认K，于确认日收盘价买入。"""
        es = "确认K收盘买入"
        if idx + 1 >= len(df):
            return self._empty_trade(signal, code, nm, '无次日数据', es)
        if not check_signal_k_confirmation(df, idx, False):
            return self._empty_trade(signal, code, nm, '确认K未满足', es)
        next_day = df.iloc[idx + 1]
        entry_price = float(next_day['close'])
        entry_date = str(next_day['date'])[:10]
        max_idx = min(idx + 1 + CONFIG['hold_days_max'], len(df))
        return self._exit_loop(df, signal, entry_price, entry_date, idx, idx + 2, max_idx, es)

    def _simulate_long_shadow_open_break_signal_close(
        self, signal: Dict, df: pd.DataFrame, idx: int, code: str, nm: str,
    ) -> Trade:
        """
        策略C（仅长下影线）：信号日收盘价 + offset；次日开盘高于该价则按次日开盘价买入。
        """
        off = float(CONFIG.get('breakout_above_signal_close_offset', 0.05))
        es = f"次日开盘破信号收盘+{off:g}"
        if signal.get("signal_type") != "长下影线":
            return self._empty_trade(signal, code, nm, '策略C仅适用长下影线', es)
        if idx + 1 >= len(df):
            return self._empty_trade(signal, code, nm, '无次日数据', es)
        next_day = df.iloc[idx + 1]
        trigger = float(signal['signal_price']) + off
        if float(next_day['open']) <= trigger:
            return self._empty_trade(signal, code, nm, '开盘未达标放弃', es)
        entry_price = float(next_day['open'])
        entry_date = str(next_day['date'])[:10]
        max_idx = min(idx + 1 + CONFIG['hold_days_max'], len(df))
        return self._exit_loop(df, signal, entry_price, entry_date, idx, idx + 2, max_idx, es)

    def _simulate_next_day_breakout(self, signal: Dict, df: pd.DataFrame, idx: int, code: str, nm: str) -> Trade:
        """非长下影信号（如阳包阴）：次日开盘突破信号日前高+0.05 则买入。"""
        es = "隔日突破买入"
        if idx + 1 >= len(df):
            return self._empty_trade(signal, code, nm, '未成交', es)

        next_day = df.iloc[idx + 1]
        trigger_price = signal['signal_high'] + 0.05

        if next_day['open'] <= trigger_price:
            return self._empty_trade(signal, code, nm, '开盘未达标放弃', es)

        entry_price = float(next_day['open'])
        entry_date = str(next_day['date'])[:10]
        max_idx = min(idx + 1 + CONFIG['hold_days_max'], len(df))
        return self._exit_loop(df, signal, entry_price, entry_date, idx, idx + 2, max_idx, es)

    def _simulate_signal_close(self, signal: Dict, df: pd.DataFrame, idx: int, code: str, nm: str) -> Trade:
        es = (
            "长下影当日收盘买入"
            if signal.get("signal_type") == "长下影线"
            else "信号日收盘买入"
        )
        row = df.iloc[idx]
        entry_price = float(row['close'])
        entry_date = str(row['date'])[:10]

        if idx + 1 >= len(df):
            return Trade(
                code, nm, signal['signal_date'], entry_date, '',
                signal['signal_price'], entry_price, 0, signal['signal_type'],
                '数据不足', 0, 0, 0, es,
            )

        max_idx = min(idx + 1 + CONFIG['hold_days_max'], len(df))
        return self._exit_loop(df, signal, entry_price, entry_date, idx, idx + 1, max_idx, es)

    def _create_trade(
        self, signal, entry_price, entry_date, exit_day,
        hold_days, exit_price, exit_reason, entry_style: str = "",
    ) -> Trade:
        pnl = exit_price - entry_price
        return Trade(
            code=signal['code'], name=signal.get('name', signal['code']),
            signal_date=signal['signal_date'],
            entry_date=entry_date, exit_date=str(exit_day['date'])[:10],
            signal_price=signal['signal_price'],
            entry_price=entry_price, exit_price=exit_price,
            signal_type=signal['signal_type'], exit_reason=exit_reason,
            pnl=pnl, pnl_pct=pnl / entry_price, hold_days=hold_days,
            entry_style=entry_style,
        )
    
    def plot_chart(self, signal: Dict, trade: Trade, df: pd.DataFrame):
        if not MPF_AVAILABLE or self.plotted_count >= CONFIG['max_charts']:
            return None
        try:
            idx = signal['signal_idx']
            chart_bars = CONFIG['chart_bars']
            start_idx = max(0, idx - chart_bars)
            end_idx = min(len(df), idx + chart_bars + 1)
            
            plot_df = df.iloc[start_idx:end_idx].copy()
            plot_df['date'] = pd.to_datetime(plot_df['date'])
            plot_df.set_index('date', inplace=True)
            
            apds = []
            entry_date = pd.to_datetime(trade.entry_date) if trade.entry_date else None
            exit_date = pd.to_datetime(trade.exit_date) if trade.exit_date else None
            
            if entry_date and entry_date in plot_df.index:
                buy_data = [trade.entry_price if d == entry_date else np.nan for d in plot_df.index]
                apds.append(mpf.make_addplot(buy_data, type='scatter', markersize=150, marker='^', color='lime'))
            
            if exit_date and exit_date in plot_df.index:
                color = 'green' if trade.pnl_pct > 0 else 'red'
                sell_data = [trade.exit_price if d == exit_date else np.nan for d in plot_df.index]
                apds.append(mpf.make_addplot(sell_data, type='scatter', markersize=150, marker='v', color=color))
            
            for p in CONFIG.get('ma_periods', [5, 10, 20, 30])[:3]:
                col = _ma_col(int(p))
                if col in plot_df.columns:
                    apds.append(mpf.make_addplot(plot_df[col], width=1))
            
            pnl_str = f"{trade.pnl_pct*100:+.1f}%"
            disp_name = signal.get('name') or trade.name or trade.code
            tag = getattr(trade, 'entry_style', '') or ''
            title = f"{tag} {disp_name} {trade.code} {trade.signal_date} {trade.signal_type} {trade.exit_reason} {pnl_str}"
            filename = _chart_filename(trade, signal)
            filepath = os.path.join(self.charts_dir, filename)
            
            fig, axes = mpf.plot(
                plot_df, type='candle', style='charles',
                title=title[:60], ylabel='价格', volume=True,
                addplot=apds, returnfig=True, figsize=(14, 8),
                panel_ratios=(3, 1), tight_layout=True
            )
            plt.savefig(filepath, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            self.plotted_count += 1
            return filepath
        except Exception:
            return None


# ==================== 报告生成 ====================
_SKIP_EXIT = frozenset(
    {
        '未成交', '限价未成交', '开盘未达标放弃', '数据不足', '确认K未满足', '无次日数据',
        '策略C仅适用长下影线',
    }
)


def _trade_rows_for_excel(trades: List[Trade]) -> List[Dict]:
    rows = []
    for t in trades:
        rows.append({
            '入场方式': getattr(t, 'entry_style', '') or '',
            '股票代码': t.code,
            '信号日期': t.signal_date,
            '信号类型': t.signal_type,
            '买入日期': t.entry_date if t.entry_price > 0 else '',
            '买入价格': round(t.entry_price, 2) if t.entry_price > 0 else '',
            '卖出日期': t.exit_date if t.exit_price > 0 else '',
            '卖出价格': round(t.exit_price, 2) if t.exit_price > 0 else '',
            '盈亏比例(%)': round(t.pnl_pct * 100, 2),
            '持仓天数': t.hold_days,
            '退出原因': t.exit_reason,
        })
    return rows


def _closed_for_stats(trades: List[Trade]) -> List[Trade]:
    """有完整买卖的成交（用于胜率）"""
    out = []
    for t in trades:
        if t.exit_reason in _SKIP_EXIT:
            continue
        if t.entry_price <= 0 or t.exit_price <= 0:
            continue
        out.append(t)
    return out


def _print_group_stats(label: str, trades: List[Trade], total_signals: int):
    closed = _closed_for_stats(trades)
    print(f"\n   【{label}】")
    print(f"      总信号: {total_signals} | 有效成交笔数: {len(closed)}")
    if not closed:
        print(f"      ⚠️ 无有效成交，无法计算胜率")
        return 0.0, 0.0
    wins = [t for t in closed if t.pnl_pct > 0]
    losses = [t for t in closed if t.pnl_pct <= 0]
    wr = len(wins) / len(closed) * 100
    total_ret = sum(t.pnl_pct for t in closed) * 100
    print(f"      盈利: {len(wins)} | 亏损: {len(losses)} | 胜率: {wr:.2f}%")
    print(f"      累计盈亏(简单加总%): {total_ret:.2f}%")
    return wr, total_ret


def generate_report_abc(
    trades_a: List[Trade],
    trades_b: List[Trade],
    trades_c: List[Trade],
    signals: List[Dict],
    output_dir: str,
):
    """对照组：策略 A / B / C，对比胜率并导出 Excel。"""
    n_sig = len(signals)
    off = float(CONFIG.get('breakout_above_signal_close_offset', 0.05))
    print(f"\n{'='*80}")
    print(f"📈 回测结果（对照组 A/B/C）")
    print(f"{'='*80}")
    print(f"   总信号数: {n_sig}")
    print(f"   A组：长下影→隔日确认K收盘；其它信号→次日开盘突破 signal_high+0.05")
    print(f"   B组：长下影→信号日收盘；其它→信号日收盘")
    print(f"   C组：仅长下影→次日开盘 > 信号日收盘+{off:.2f} 则开盘价入；其它类型记「策略C仅适用长下影线」")

    wr_a, ret_a = _print_group_stats("A 策略A(确认K收盘/前高突破)", trades_a, n_sig)
    wr_b, ret_b = _print_group_stats("B 策略B(当日收盘)", trades_b, n_sig)
    wr_c, ret_c = _print_group_stats("C 策略C(开盘破收盘+offset)", trades_c, n_sig)

    print(f"\n   【胜率对比】")
    print(f"      A: {wr_a:.2f}%  |  B: {wr_b:.2f}%  |  C: {wr_c:.2f}%")
    print(f"      B-A: {wr_b - wr_a:+.2f}%  |  C-A: {wr_c - wr_a:+.2f}%  |  C-B: {wr_c - wr_b:+.2f}%")

    na, nb, nc = len(_closed_for_stats(trades_a)), len(_closed_for_stats(trades_b)), len(_closed_for_stats(trades_c))
    try:
        excel_path = os.path.join(output_dir, f'交易记录_{CONFIG["start_date"]}_{CONFIG["end_date"]}.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(_trade_rows_for_excel(trades_a)).to_excel(writer, sheet_name='A_策略A', index=False)
            pd.DataFrame(_trade_rows_for_excel(trades_b)).to_excel(writer, sheet_name='B_策略B', index=False)
            pd.DataFrame(_trade_rows_for_excel(trades_c)).to_excel(writer, sheet_name='C_策略C', index=False)
            pd.DataFrame([
                {
                    '指标': '胜率(%)',
                    'A_策略A': round(wr_a, 4), 'B_策略B': round(wr_b, 4), 'C_策略C': round(wr_c, 4),
                    'B减A': round(wr_b - wr_a, 4), 'C减A': round(wr_c - wr_a, 4), 'C减B': round(wr_c - wr_b, 4),
                },
                {
                    '指标': '累计盈亏加总(%)',
                    'A_策略A': round(ret_a, 4), 'B_策略B': round(ret_b, 4), 'C_策略C': round(ret_c, 4),
                    'B减A': round(ret_b - ret_a, 4), 'C减A': round(ret_c - ret_a, 4), 'C减B': round(ret_c - ret_b, 4),
                },
                {
                    '指标': '有效成交笔数',
                    'A_策略A': na, 'B_策略B': nb, 'C_策略C': nc,
                    'B减A': nb - na, 'C减A': nc - na, 'C减B': nc - nb,
                },
            ]).to_excel(writer, sheet_name='对照汇总', index=False)
        print(f"\n💾 Excel已导出: {excel_path}")
    except Exception as e:
        print(f"\n⚠️ Excel导出失败: {e}")


# ==================== 主程序 ====================
def main():
    _ensure_utf8_stdio()
    load_stock_names()
    print(f"{'='*80}")
    print(f"通达信选股策略回测 - 长下影线(锤子线)版")
    print(f"{'='*80}")
    
    if not os.path.exists(CONFIG['zip_file']):
        print(f"❌ 找不到数据文件: {CONFIG['zip_file']}")
        return
    
    print(f"\n📦 扫描数据...")
    all_files = []
    with zipfile.ZipFile(CONFIG['zip_file'], 'r') as zf:
        all_files = [f for f in zf.namelist() if f.endswith(('.day', '.csv'))]
    
    filtered_stocks = []
    for path in all_files:
        basename = os.path.basename(path)
        raw = basename.split('.')[0]  # 如 sh600000

        if raw.lower().startswith(('sh', 'sz')) and len(raw) >= 8 and raw[2:8].isdigit():
            code = raw[2:8]
        else:
            code = ''.join(c for c in raw if c.isdigit())[:6]
        if len(code) == 6 and filter_stock_code(code):
            filtered_stocks.append((code, path))
    
    print(f"   找到 {len(filtered_stocks)} 只股票")
    
    start_date = str(CONFIG['start_date'])
    end_date = str(CONFIG['end_date'])
    
    charts_output_dir = os.path.join(CONFIG['charts_dir'], f"{start_date}_{end_date}")
    os.makedirs(charts_output_dir, exist_ok=True)
    
    signals = screen_stocks(CONFIG['zip_file'], filtered_stocks, start_date, end_date)
    
    if not signals:
        print(f"\n❌ 未找到信号（请检查：CONFIG['zip_file'] 是否存在、start_date/end_date 是否与数据重叠、filter_stock_code 是否过严）")
        return
    
    print(f"\n🔄 回测 {len(signals)} 个信号（A + B + C 对照）...")
    simulator = TradeSimulator(charts_dir=charts_output_dir)
    trades_next: List[Trade] = []
    trades_close: List[Trade] = []
    trades_break_close: List[Trade] = []
    data_cache = {}
    
    for i, sig in enumerate(signals):
        code = sig['code']
        file_path = sig['file_path']
        
        if file_path not in data_cache:
            df = read_stock_from_zip(CONFIG['zip_file'], file_path)
            if df is not None:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = calculate_ma(df)
                data_cache[file_path] = df
        else:
            df = data_cache[file_path]
        
        if df is not None:
            t_a = simulator.simulate(sig, df, TradeSimulator.ENTRY_STRATEGY_A)
            t_b = simulator.simulate(sig, df, TradeSimulator.ENTRY_SIGNAL_CLOSE)
            t_c = simulator.simulate(sig, df, TradeSimulator.ENTRY_STRATEGY_C)
            trades_next.append(t_a)
            trades_close.append(t_b)
            trades_break_close.append(t_c)
            ok_a = t_a.exit_reason not in _SKIP_EXIT and t_a.entry_price > 0
            ok_b = t_b.exit_reason not in _SKIP_EXIT and t_b.entry_price > 0
            ok_c = t_c.exit_reason not in _SKIP_EXIT and t_c.entry_price > 0
            print(
                f"   [{i+1}] {code} | A:{t_a.exit_reason} {t_a.pnl_pct*100:+.2f}% | "
                f"B:{t_b.exit_reason} {t_b.pnl_pct*100:+.2f}% | C:{t_c.exit_reason} {t_c.pnl_pct*100:+.2f}%"
            )
            if ok_a:
                simulator.plot_chart(sig, t_a, df)
            if ok_b:
                simulator.plot_chart(sig, t_b, df)
            if ok_c:
                simulator.plot_chart(sig, t_c, df)
    
    if CONFIG['zip_file'] in _zip_cache:
        _zip_cache[CONFIG['zip_file']].close()
        del _zip_cache[CONFIG['zip_file']]
    
    generate_report_abc(trades_next, trades_close, trades_break_close, signals, charts_output_dir)
    print(f"\n✅ 完成! 结果在: {charts_output_dir}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
