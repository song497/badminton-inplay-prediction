"""Parse the raw ShuttleSet releases into a single stroke-level table.

Input : raw/set/<match>/set*.csv   (ShuttleSet and ShuttleSet22 releases)
Output: data/dataset_blsr_all.csv   one row per stroke
        data/game_to_match.csv      game_id -> match_id mapping
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_ROOT = str(ROOT / "raw")
OUTPUT_CSV = str(DATA / "dataset_blsr_all.csv")
GAME_MAP_FILE = str(DATA / "game_to_match.csv")   # 保存 game_id, match_id, source_file 映射

# 动作映射（保留原始映射，缺失的用 unknown）
ch_to_en = {
    '發長球': 'long service',
    '發短球': 'short service',
    '殺球': 'smash',
    '點扣': 'wrist smash',
    '撲球': 'rush',
    '切球': 'drop',
    '過度切球': 'passive drop',
    '挑球': 'lob',
    '防守回挑': 'defensive return lob',
    '推球': 'push',
    '放小球': 'net shot',
    '勾球': 'cross-court net shot',
    '擋小球': 'return net',
    '長球': 'clear',
    '平球': 'drive',
    '後場抽平球': 'back-court drive',
    '防守回抽': 'defensive return drive',
    '未知球種': 'unknown',
    '小平球': 'drive',
    '擋網': 'block',
    '平抽球': 'drive',
}

def get_label(df, rally_id, server):
    rally = df[df['rally'] == rally_id].sort_values('ball_round')
    if len(rally) < 2:
        return 0
    first = rally.iloc[0]
    last = rally.iloc[-1]
    if last['roundscore_A'] > first['roundscore_A']:
        winner = 'A'
    elif last['roundscore_B'] > first['roundscore_B']:
        winner = 'B'
    else:
        if 'getpoint_player' in last and pd.notna(last['getpoint_player']):
            winner = last['getpoint_player']
        elif 'win_reason' in last and pd.notna(last['win_reason']):
            winner = last['player']
        elif 'lose_reason' in last and pd.notna(last['lose_reason']):
            winner = 'A' if last['player'] == 'B' else 'B'
        else:
            return 0
    return 1 if winner == server else 0

def process_game(filepath, match_id, game_id, rally_counter, records):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filepath, encoding='gbk')
        except Exception as e:
            print(f"  ⚠️ 读取失败: {filepath} - {e}")
            return rally_counter
    required = ['rally', 'ball_round', 'player', 'type', 'roundscore_A', 'roundscore_B']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ 缺少列 {missing}: {filepath}")
        return rally_counter
    # 填充缺失列
    for col in ['hit_area', 'player_location_area', 'opponent_location_area']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)
    if 'aroundhead' not in df.columns:
        df['aroundhead'] = 0
    else:
        df['aroundhead'] = df['aroundhead'].fillna(0).astype(int)
    if 'backhand' not in df.columns:
        df['backhand'] = 0
    else:
        df['backhand'] = df['backhand'].fillna(0).astype(int)
    # 映射技术
    df['type_en'] = df['type'].map(ch_to_en).fillna('unknown')
    # 发球方
    first = df[df['ball_round'] == 1.0].iloc[0]
    server = first['player']
    # 逐回合处理
    for rid in df['rally'].unique():
        rally = df[df['rally'] == rid].sort_values('ball_round')
        if len(rally) < 2:
            continue
        label = get_label(df, rid, server)
        first_row = rally.iloc[0]
        score_diff = first_row['roundscore_A'] - first_row['roundscore_B']
        n_shots = len(rally)
        for _, row in rally.iterrows():
            ball_round = int(row['ball_round'])
            if n_shots > 1:
                time_prop = (ball_round - 1) / (n_shots - 1)
            else:
                time_prop = 0.0
            time_prop = np.clip(time_prop, 0.0, 1.0)
            is_target = 1 if row['player'] == server else 0
            records.append({
                'game_id': game_id,
                'rally_id': rally_counter,
                'match_id': match_id,
                'is_target_turn': is_target,
                'aroundhead': int(row['aroundhead']),
                'backhand': int(row['backhand']),
                'time_proportion': time_prop,
                'roundscore_diff': score_diff,
                'continuous_score': 0,
                'hit_area': int(row['hit_area']),
                'player_location_area': int(row['player_location_area']),
                'opponent_location_area': int(row['opponent_location_area']),
                'type': row['type_en'],
                'is_target_win': label,
            })
        rally_counter += 1
    return rally_counter

def main():
    # 遍历 set 和 set22 下的所有子文件夹（真正的比赛文件夹）
    match_folders = []
    for dataset in ['set', 'set22']:
        dataset_path = os.path.join(RAW_ROOT, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for match_name in os.listdir(dataset_path):
            match_path = os.path.join(dataset_path, match_name)
            if os.path.isdir(match_path):
                match_folders.append(match_path)
    print(f"发现 {len(match_folders)} 个比赛文件夹")
    if not match_folders:
        print("错误：未找到比赛文件夹，请检查 RAW_ROOT 路径")
        return

    all_records = []
    rally_counter = 1
    match_id = 1
    game_id = 1
    game_map = []   # 存储 (game_id, match_id, source_file)

    for folder in match_folders:
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"⚠️ 比赛文件夹无 CSV: {folder}")
            continue
        print(f"处理比赛 {match_id}: {os.path.basename(folder)} ({len(csv_files)} 局)")
        for csv_file in csv_files:
            filepath = os.path.join(folder, csv_file)
            # 记录映射
            game_map.append([game_id, match_id, os.path.relpath(filepath, RAW_ROOT)])
            rally_counter = process_game(filepath, match_id, game_id, rally_counter, all_records)
            game_id += 1
        match_id += 1

    if not all_records:
        print("未生成任何数据")
        return

    df_out = pd.DataFrame(all_records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"生成 {len(df_out)} 行记录")
    print(f"比赛数: {df_out['match_id'].nunique()}")
    print(f"局数: {df_out['game_id'].nunique()}")
    print(f"正样本比例: {df_out['is_target_win'].mean():.3f}")

    # 保存 game_id 映射
    df_map = pd.DataFrame(game_map, columns=['game_id', 'match_id', 'source_file'])
    df_map.to_csv(GAME_MAP_FILE, index=False)
    print(f"局映射保存至 {GAME_MAP_FILE}")

if __name__ == '__main__':
    main()