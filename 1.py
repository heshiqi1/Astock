#!/usr/bin/env python3
"""
通达信选股策略回测 - 严格长下影线(倒锤子)优化版

长下影线(实为倒锤子线)条件：
1. 回调位置：最低点低于前3根K线最低点
2. 不大涨：当日涨幅<5%
3. 形态：上影线>=实体(占1/2或1/3)，下影线<=实体*0.2，收阳(与下跌趋势相反)
4. 重叠检查：当前实体与前一根实体重叠<30%
5. 确认K线：下一根是强阳线(非十字星，实体>前一根实体)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
import zipfile
import struct
import warnings
import multiprocessing

warnings.filterwarnings('ignore')

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
    'max_charts': 2000,
    'chart_bars': 100,
    
    'target_date': None,
    'start_date': '20200101',
    'end_date': '20220331',
    
    'hold_days_max': 60,
    'take_profit_pct': 0.15,
    'stop_loss_pct': 0.03,
    'use_trailing_stop': True,
    'trailing_stop_pct': 0.10,
    
    'debug_mode': True,
    'debug_max_detail': 5,
}


# ==================== 代码过滤 ====================
def get_market_name(code: str) -> str:
    if not code or len(code) != 6:
        return 'invalid'
    if code.startswith(('600', '601', '603', '605')):
        return 'sh_main'
    if code.startswith(('000', '001', '002', '003')):
        return 'sz_main'
    if code.startswith(('300', '301')):
        return 'sz_cyb'
    if code.startswith('688'):
        return 'sh_kcb'
    if code in ('4', '8'):
        return 'bj'
    return 'other'


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


# ==================== 数据读取 ====================
def parse_day_binary(data: bytes) -> Optional[pd.DataFrame]:
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
        record = struct.unpack_from('IIIIIIII', data, offset)
        date_int = record
        dates[i] = np.datetime64(f"{date_int//10000:04d}-{(date_int%10000)//100:02d}-{date_int%100:02d}")
        opens[i] = record / 100.0
        highs[i] = record / 100.0
        lows[i] = record / 100.0
        closes[i] = record / 100.0
        volumes[i] = record
    
    return pd.DataFrame({
        'date': dates, 'open': opens, 'high': highs,
        'low': lows, 'close': closes, 'volume': volumes
    })


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
    except Exception:
        pass
    return None


# ==================== 技术指标 ====================
def calculate_ma(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close'].values
    df['MA5'] = pd.Series(close).rolling(5, min_periods=1).mean().values
    df['MA10'] = pd.Series(close).rolling(10, min_periods=1).mean().values
    df['MA20'] = pd.Series(close).rolling(20, min_periods=1).mean().values
    df['MA30'] = pd.Series(close).rolling(30, min_periods=1).mean().values
    return df


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    if len(prices) < 2:
        return np.array( * len(prices))
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gains = pd.Series(gains).rolling(window=period, min_periods=1).mean().values
    avg_losses = pd.Series(losses).rolling(window=period, min_periods=1).mean().values
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[30], rsi])


# ==================== 【核心】改进的信号检测 ====================
def check_signal_advanced(df: pd.DataFrame, valid_indices: np.ndarray, code: str, show_detail: bool = False) -> List[Tuple[int, str]]:
    """改进版信号检测（严格长下影线/倒锤子线 + 阳包阴）"""
    signals = []
    
    if len(df) < 35:
        return signals
    
    df = df.reset_index(drop=True)
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    ma5, ma10, ma20, ma30 = df['MA5'].values, df['MA10'].values, df['MA20'].values, df['MA30'].values
    rsi = calculate_rsi(close)
    
    # 多头发散条件（用于阳包阴）
    bullish_trend = (close > ma5) & (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma30) & (rsi < 70)
    
    first_detail_shown = False
    
    for idx in valid_indices:
        idx = int(idx)
        if idx < 5 or idx >= len(df) - 1:  # 需要前3根+后1根确认
            continue
        
        # 显示第一条数据的调试信息
        if show_detail and not first_detail_shown and idx >= 30:
            row = df.iloc[idx]
            print(f"\n  🔍 {code} 检查点 (索引{idx}, {row['date']}):")
            print(f"     价格: O{row['open']:.2f} H{row['high']:.2f} L{row['low']:.2f} C{row['close']:.2f}")
            first_detail_shown = True
        
        # 【策略1】严格长下影线（倒锤子线）- 回调买入
        is_long_shadow = check_strict_long_shadow(df, idx, show_detail and idx == valid_indices)
        if is_long_shadow:
            signals.append((idx, '长下影线'))
            if show_detail:
                print(f"     ✅ 长下影线信号确认")
            continue
        
        # 【策略2】阳包阴（保持原有逻辑，但放宽条件）
        if idx >= 4 and bullish_trend[idx]:
            try:
                d = df.iloc[idx-4:idx+1]
                if (len(d) == 5 and
                    d.iloc['open'] > d.iloc['close'] and  # 前1天阴线
                    d.iloc['close'] > d.iloc['open'] and  # 包络
                    d.iloc['close'] > d.iloc['open'] and  # 当天阳线
                    d.iloc['volume'] < d.iloc['volume'] * 1.5):  # 成交量不过大
                    signals.append((idx, '阳包阴'))
            except:
                pass
    
    return signals


def check_strict_long_shadow(df: pd.DataFrame, idx: int, show_detail: bool = False) -> bool:
    """
    严格长下影线（实为倒锤子线）检测：
    1. 回调：low < min(前3根low)
    2. 不大涨：(close-open)/open < 5%
    3. 形态：上影线>=实体(占1/2或1/3)，下影线<=实体*0.2，收阳
    4. 重叠：当前实体与前实体重叠<30%
    5. 确认：下一根强阳线（非十字星，实体>当前实体）
    """
    if idx < 4 or idx >= len(df) - 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    next_c = df.iloc[idx+1]
    
    # 1. 回调位置：当前low低于前3根low的最低点
    prev3_lows = df.iloc[idx-3:idx]['low'].values
    if curr['low'] >= prev3_lows.min() * 0.999:  # 允许微小误差
        if show_detail:
            print(f"     ❌ 非回调: 当前low{curr['low']:.2f} >= 前3低{prev3_lows.min():.2f}")
        return False
    
    # 2. 不能大涨 (<5%)
    change_pct = (curr['close'] - curr['open']) / curr['open']
    if change_pct > 0.05:
        if show_detail:
            print(f"     ❌ 大涨: {change_pct*100:.1f}% > 5%")
        return False
    
    # 3. 形态计算
    body = abs(curr['close'] - curr['open'])
    upper_shadow = curr['high'] - max(curr['close'], curr['open'])
    lower_shadow = min(curr['close'], curr['open']) - curr['low']
    
    if body < 0.01:  # 避免除零，实体不能太小
        if show_detail:
            print(f"     ❌ 实体太小")
        return False
    
    # 上影线占1/2或1/3（即上影线 >= 实体长度）
    if upper_shadow < body * 0.8:  # 至少0.8倍实体，接近1倍
        if show_detail:
            print(f"     ❌ 上影线不够长: {upper_shadow:.2f} < {body*0.8:.2f} (实体{body:.2f})")
        return False
    
    # 下影线很短 (<20%实体)
    if lower_shadow > body * 0.2:
        if show_detail:
            print(f"     ❌ 下影线太长: {lower_shadow:.2f} > {body*0.2:.2f}")
        return False
    
    # 收阳（与下跌趋势相反）
    if curr['close'] <= curr['open']:
        if show_detail:
            print(f"     ❌ 非阳线")
        return False
    
    # 4. 重叠检查：当前实体与前一根实体重叠<30%
    curr_body_top = max(curr['open'], curr['close'])
    curr_body_bottom = min(curr['open'], curr['close'])
    prev_body_top = max(prev['open'], prev['close'])
    prev_body_bottom = min(prev['open'], prev['close'])
    
    overlap_top = min(curr_body_top, prev_body_top)
    overlap_bottom = max(curr_body_bottom, prev_body_bottom)
    overlap = max(0, overlap_top - overlap_bottom)
    
    if body > 0 and overlap / body > 0.3:  # 重叠超过30%
        if show_detail:
            print(f"     ❌ 重叠太多: {overlap/body*100:.0f}% > 30%")
        return False
    
    # 5. 确认K线检查：下一根是强入场K线
    next_body = abs(next_c['close'] - next_c['open'])
    next_upper = next_c['high'] - max(next_c['close'], next_c['open'])
    next_lower = min(next_c['close'], next_c['open']) - next_c['low']
    next_shadow = next_upper + next_lower
    
    # 非十字星（实体 > 影线）
    if next_body <= next_shadow * 0.8:
        if show_detail:
            print(f"     ❌ 确认K线太弱(十字星): 实体{next_body:.2f} <= 影线{next_shadow:.2f}")
        return False
    
    # 确认K线是阳线
    if next_c['close'] <= next_c['open']:
        if show_detail:
            print(f"     ❌ 确认K线非阳线")
        return False
    
    # 确认K线实体较大（大于当前实体，显示强势）
    if next_body < body * 0.9:
        if show_detail:
            print(f"     ❌ 确认K线实体不够大: {next_body:.2f} < {body*0.9:.2f}")
        return False
    
    if show_detail:
        print(f"     ✔️ 上影线{upper_shadow:.2f}>=实体{body:.2f}, 重叠{overlap/body*100:.0f}%, 确认K线实体{next_body:.2f}")
    
    return True


# ==================== 筛选 ====================
def screen_stocks(zip_path: str, stocks: List[Tuple[str, str]], start_date: str, end_date: str) -> List[Dict]:
    signals = []
    total = len(stocks)
    
    print(f"\n{'='*80}")
    print(f"🔍 开始筛选（严格长下影线策略）")
    print(f"   股票总数: {total}")
    print(f"   回测区间: {start_date} ~ {end_date}")
    print(f"{'='*80}")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    processed = 0
    
    for i, (code, file_path) in enumerate(stocks):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"   进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) | 已发现信号: {len(signals)}")
        
        show_detail = CONFIG['debug_mode'] and processed < CONFIG['debug_max_detail']
        
        df = read_stock_from_zip(zip_path, file_path)
        if df is None or len(df) < 35:
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
        
        signal_list = check_signal_advanced(df, valid_indices, code, show_detail)
        
        for idx, signal_type in signal_list:
            row = df.iloc[idx]
            pressure = df.iloc[max(0, idx-20):idx]['high'].max() if idx > 0 else row['high']
            
            signals.append({
                'code': code,
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
        
        if show_detail:
            processed += 1
    
    if zip_path in _zip_cache:
        _zip_cache[zip_path].close()
        del _zip_cache[zip_path]
    
    print(f"\n{'='*80}")
    print(f"📊 筛选完成: 共发现 {len(signals)} 个信号")
    print(f"{'='*80}")
    
    return signals


# ==================== 交易模拟 ====================
class TradeSimulator:
    def __init__(self, charts_dir: str = None):
        self.charts_dir = charts_dir or CONFIG['charts_dir']
        self.plotted_count = 0
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)
    
    def simulate(self, signal: Dict, df: pd.DataFrame) -> Trade:
        idx = signal['signal_idx']
        code = signal['code']
        
        if idx + 1 >= len(df):
            return Trade(code, code, signal['signal_date'], '', '', 
                        signal['signal_price'], 0, 0, signal['signal_type'], 
                        '未成交', 0, 0, 0)
        
        next_day = df.iloc[idx + 1]
        trigger_price = signal['signal_high'] + 0.05
        
        if next_day['open'] <= trigger_price:
            return Trade(code, code, signal['signal_date'], '', '', 
                        signal['signal_price'], 0, 0, signal['signal_type'], 
                        '开盘未达标放弃', 0, 0, 0)
        
        entry_price = next_day['open']
        entry_date = str(next_day['date'])[:10]
        stop_loss = entry_price * (1 - CONFIG['stop_loss_pct'])
        take_profit = entry_price * (1 + CONFIG['take_profit_pct'])
        highest_price = entry_price
        pressure = signal['pressure']
        max_idx = min(idx + 1 + CONFIG['hold_days_max'], len(df))
        
        for i in range(idx + 2, max_idx):
            day = df.iloc[i]
            hold_days = i - idx - 1
            
            if day['high'] > highest_price:
                highest_price = day['high']
            
            if CONFIG['use_trailing_stop']:
                profit_pct = (highest_price - entry_price) / entry_price
                if profit_pct >= 0.10:
                    trailing_stop = highest_price * 0.95
                    if day['low'] <= trailing_stop:
                        return self._create_trade(signal, entry_price, entry_date, 
                                                day, hold_days, trailing_stop, '移动止盈')
            
            if day['open'] <= stop_loss:
                return self._create_trade(signal, entry_price, entry_date, 
                                        day, hold_days, day['open'], '止损跳空')
            if day['low'] <= stop_loss:
                return self._create_trade(signal, entry_price, entry_date, 
                                        day, hold_days, stop_loss, '止损')
            
            if day['high'] >= take_profit:
                exit_price = max(day['open'], take_profit)
                return self._create_trade(signal, entry_price, entry_date,
                                        day, hold_days, exit_price, '止盈-15%')
            
            if day['high'] >= pressure and day['high'] > entry_price * 1.03:
                exit_price = max(day['open'], pressure)
                return self._create_trade(signal, entry_price, entry_date,
                                        day, hold_days, exit_price, '止盈-压力位')
        
        last_day = df.iloc[max_idx - 1]
        return self._create_trade(signal, entry_price, entry_date,
                                last_day, max_idx - idx - 2, last_day['close'], '到期平仓')
    
    def _create_trade(self, signal, entry_price, entry_date, exit_day, 
                     hold_days, exit_price, exit_reason) -> Trade:
        pnl = exit_price - entry_price
        return Trade(
            code=signal['code'], name=signal['code'],
            signal_date=signal['signal_date'],
            entry_date=entry_date, exit_date=str(exit_day['date'])[:10],
            signal_price=signal['signal_price'],
            entry_price=entry_price, exit_price=exit_price,
            signal_type=signal['signal_type'], exit_reason=exit_reason,
            pnl=pnl, pnl_pct=pnl/entry_price, hold_days=hold_days
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
            
            signal_date = pd.to_datetime(trade.signal_date)
            entry_date = pd.to_datetime(trade.entry_date) if trade.entry_date else None
            exit_date = pd.to_datetime(trade.exit_date) if trade.exit_date else None
            
            if entry_date and entry_date in plot_df.index:
                buy_data = [trade.entry_price if d == entry_date else np.nan for d in plot_df.index]
                apds.append(mpf.make_addplot(buy_data, type='scatter', markersize=150, marker='^', color='lime'))
            
            if exit_date and exit_date in plot_df.index:
                color = 'green' if trade.pnl_pct > 0 else 'red'
                sell_data = [trade.exit_price if d == exit_date else np.nan for d in plot_df.index]
                apds.append(mpf.make_addplot(sell_data, type='scatter', markersize=150, marker='v', color=color))
            
            for ma in [5, 10, 20]:
                if f'MA{ma}' in plot_df.columns:
                    apds.append(mpf.make_addplot(plot_df[f'MA{ma}'], width=1))
            
            pnl_str = f"{trade.pnl_pct*100:+.1f}%"
            title = f"{trade.code} {trade.signal_date} {trade.signal_type} {trade.exit_reason} {pnl_str}"
            filename = f"{trade.code}_{trade.signal_date}_{trade.exit_reason}.png"
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
def generate_report(trades: List[Trade], signals: List[Dict], output_dir: str):
    closed = [t for t in trades if t.exit_reason not in ('未成交', '限价未成交', '开盘未达标放弃')]
    
    print(f"\n{'='*80}")
    print(f"📈 回测结果")
    print(f"{'='*80}")
    print(f"   总信号: {len(signals)} | 成交: {len(closed)}")
    
    if not closed:
        print("   ⚠️ 无成交记录")
        export_to_excel(trades, signals, output_dir)
        return
    
    wins = [t for t in closed if t.pnl_pct > 0]
    losses = [t for t in closed if t.pnl_pct <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    
    print(f"   盈利: {len(wins)} | 亏损: {len(losses)} | 胜率: {win_rate:.1f}%")
    print(f"   总收益率: {sum([t.pnl_pct for t in closed]) * 100:.2f}%")
    
    reasons = {}
    for t in closed:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\n   退出原因:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x):
        print(f"      {reason}: {count}笔")
    
    export_to_excel(trades, signals, output_dir)


def export_to_excel(trades: List[Trade], signals: List[Dict], output_dir: str):
    try:
        all_trade_data = []
        for t in trades:
            all_trade_data.append({
                '股票代码': t.code,
                '信号日期': t.signal_date,
                '信号类型': t.signal_type,
                '信号价格': round(t.signal_price, 2),
                '买入日期': t.entry_date if t.entry_price > 0 else '',
                '买入价格': round(t.entry_price, 2) if t.entry_price > 0 else '',
                '卖出日期': t.exit_date if t.exit_price > 0 else '',
                '卖出价格': round(t.exit_price, 2) if t.exit_price > 0 else '',
                '盈亏金额': round(t.pnl, 2),
                '盈亏比例(%)': round(t.pnl_pct * 100, 2),
                '持仓天数': t.hold_days,
                '退出原因': t.exit_reason,
                '状态': '成交' if t.exit_reason not in ('未成交', '限价未成交', '开盘未达标放弃') else '未成交'
            })
        
        df_all = pd.DataFrame(all_trade_data)
        excel_path = os.path.join(output_dir, f'交易记录_{CONFIG["start_date"]}_{CONFIG["end_date"]}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_all.to_excel(writer, sheet_name='全部交易', index=False)
            df_closed = df_all[df_all['状态'] == '成交'].copy()
            if len(df_closed) > 0:
                df_closed.to_excel(writer, sheet_name='成交记录', index=False)
            
            closed = [t for t in trades if t.exit_reason not in ('未成交', '限价未成交', '开盘未达标放弃')]
            wins = [t for t in closed if t.pnl_pct > 0]
            summary_data = {
                '指标': ['总信号数', '成交笔数', '盈利笔数', '亏损笔数', '胜率(%)', '总收益率(%)'],
                '数值': [
                    len(signals), len(closed), len(wins), len(closed) - len(wins),
                    round(len(wins) / len(closed) * 100, 2) if closed else 0,
                    round(sum([t.pnl_pct for t in closed]) * 100, 2) if closed else 0
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='统计汇总', index=False)
        
        print(f"\n💾 Excel已导出: {excel_path}")
    except Exception as e:
        print(f"\n⚠️ Excel导出失败: {e}")


# ==================== 主程序 ====================
def main():
    print(f"{'='*80}")
    print(f"通达信选股策略回测 - 严格长下影线(倒锤子)版")
    print(f"策略要点:")
    print(f"  1. 回调位置: low < min(前3根low)")
    print(f"  2. 形态: 上影线>=实体(占1/2或1/3), 下影线<=实体*0.2, 收阳")
    print(f"  3. 重叠: 实体与前根重叠<30%")
    print(f"  4. 确认: 下一根强阳线(非十字星, 实体>当前)")
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
        raw = basename.split('.')
        if raw.lower().startswith(('sh', 'sz')) and len(raw) >= 8:
            code = raw[2:8]
        else:
            code = ''.join(c for c in raw if c.isdigit())[:6]
        if len(code) == 6 and filter_stock_code(code):
            filtered_stocks.append((code, path))
    
    print(f"   找到 {len(filtered_stocks)} 只股票")
    
    if CONFIG.get('target_date'):
        start_date = end_date = CONFIG['target_date']
    else:
        start_date = CONFIG['start_date']
        end_date = CONFIG['end_date']
    
    charts_output_dir = os.path.join(CONFIG['charts_dir'], f"{start_date}_{end_date}")
    os.makedirs(charts_output_dir, exist_ok=True)
    
    signals = screen_stocks(CONFIG['zip_file'], filtered_stocks, start_date, end_date)
    
    if not signals:
        print(f"\n❌ 未找到信号，建议:")
        print(f"   1. 放宽'上影线>=实体'条件（当前0.8倍）")
        print(f"   2. 放宽'回调'定义（当前要求low<前3根最低）")
        print(f"   3. 检查数据日期范围")
        return
    
    print(f"\n🔄 回测 {len(signals)} 个信号...")
    simulator = TradeSimulator(charts_dir=charts_output_dir)
    trades = []
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
            trade = simulator.simulate(sig, df)
            trades.append(trade)
            status = "✅" if trade.exit_reason not in ('未成交', '限价未成交', '开盘未达标放弃') else "⚠️"
            print(f"   [{i+1}] {status} {trade.code} | {trade.exit_reason} | {trade.pnl_pct*100:+.2f}%")
            simulator.plot_chart(sig, trade, df)
    
    if CONFIG['zip_file'] in _zip_cache:
        _zip_cache[CONFIG['zip_file']].close()
        del _zip_cache[CONFIG['zip_file']]
    
    generate_report(trades, signals, charts_output_dir)
    print(f"\n✅ 完成! 结果在: {charts_output_dir}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
