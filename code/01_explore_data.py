import sys

import numpy as np

from hfr_utils import DATA_FILE, load_hfr_data


# 展示帧数量来源：避免控制台输出过长
MAX_DISPLAY_FRAME_COUNT = 10


sys.stdout.reconfigure(encoding='utf-8')

hfr = load_hfr_data()
print(f'数据文件：{DATA_FILE}')
print(f'HFRData 形状: {hfr.shape}')
print(f'帧数量: {hfr.shape[1]}')
print()

# 查看第一帧
f0 = hfr[0, 0]
names = list(f0.dtype.names)
print(f'第一帧字段 ({len(names)}个): {names}')
print()

for fn in names:
    val = f0[0, 0][fn]
    try:
        if isinstance(val, np.ndarray):
            data = val.ravel()
            print(f'  {fn}: shape={val.shape}, 非零数={np.count_nonzero(data)}/{data.size}, 样例: {data[:5]}')
        else:
            print(f'  {fn}: type={type(val)}')
    except Exception as e:
        print(f'  {fn}: type={type(val)}, err={e}')

# 统计所有帧
print()
print('=== 各帧统计 ===')
for i in range(min(hfr.shape[1], MAX_DISPLAY_FRAME_COUNT)):
    frame = hfr[0, i]
    pc = frame[0, 0]['PlotCnt']
    nid = frame[0, 0]['id']
    if isinstance(nid, np.ndarray):
        npc = nid.ravel().shape[0]
    else:
        npc = 0
    print(f'帧{i+1}: PlotCnt={pc}, id数量={npc}')
