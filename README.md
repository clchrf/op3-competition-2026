# 學長姐 2026 夏天比賽程式（competition_2026）

> 2026-10-13 從隨身碟（E:）整理。學長說這是**去年比賽用的程式**，分別在**不同的 workspace**，
> 而且**同名的東西在不同 workspace 裡參數不一樣**（這可能是關鍵）。
> 這裡是**參考複本**：隨身碟上的原檔沒有動；複本的程式碼每一行都和原檔一樣，
> 只多了開頭說明 + `# NOTE(cindy):` 註解行，並把 CRLF 換成 LF。
> 上一層 `team_reference/` 有 `CATKIN_IGNORE`，catkin 不會編譯這裡。

## 學長指定的 4 支程式

| 項目 | 原本位置（機器人上） | 這裡 |
|---|---|---|
| Sprint | `/home/robotis/yuAn_ws/src/an_pkg/sprint_canada.py` | `sprint_yuAn/sprint_canada.py` |
| 射箭（逆時針） | `/home/robotis/T_Fira_ws/src/TT_FIRA_2024/scripts/archery_260718_COUNTER_CLOCKWISE.py` | `archery_TFira/archery_260718_COUNTER_CLOCKWISE.py` |
| 避障（主程式） | `/home/robotis/jeff_ws/src/fira_basketball/scripts/jeff_obstacle4 (copy 1).py` | `obstacle_jeff/jeff_obstacle4_copy1.py` |
| 避障（影像） | `/home/robotis/jeff_ws/src/fira_basketball/scripts/jeff_obstacle_vision (copy 1).py` | `obstacle_jeff/jeff_obstacle_vision_copy1.py` |

- 學長寫的是 `jeff_obstacle (copy 1).py`，但資料夾裡**沒有這個檔名**。實際的是 `jeff_obstacle4 (copy 1).py`，
  它跟 `jeff_obstacle_vision (copy 1).py` 是一組（topic `obstacle_vision_data` / `obstacle_head_angle` 對得上）。
  另外有一支 `jeff_obstacle.py`（舊的單一程式版）沒有複製。**要跟學長確認一下。**
- 檔名裡的空白和括號拿掉了（`rosrun` 不喜歡空白）。

每個資料夾都放了**那支程式需要的東西**（它 import 的那個版本的函式庫、它用的走路參數檔）。

```
competition_2026/
  sprint_yuAn/      sprint_canada.py + op3_ros_utils2.py (yuAn 版) + walking_config2/*.yaml (7 個)
  archery_TFira/    archery_260718_COUNTER_CLOCKWISE.py + op3_ros_utils.py (T_Fira 版)
  obstacle_jeff/    jeff_obstacle4_copy1.py + jeff_obstacle_vision_copy1.py + op3_ros_utils.py (jeff 版)
  weightlift_Ray/   舉重 op3_weightlifting_v4.py + detector.py + YOLO 模型 + op3_ros_utils.py (Ray_wk 版)
  motion_files/     4 個 workspace 的動作檔 motion_4095.bin + page_diff.txt (每一頁的比較)
  walking_params/   4 個 workspace 的 op3_walking_module/config/param.yaml
  tools/            compare_motion_files.py (比較動作檔的小工具, python 2/3 都可以)
```

## 一定要知道：每個 workspace 不一樣的地方（關鍵）

### 1. 動作檔 motion_4095.bin —— 每個 workspace 都不一樣

`op3_manager` 用的是**你 source 的那個 workspace** 裡 `op3_action_module/data/motion_4095.bin`。
同一個頁碼，在不同 workspace 是**完全不同的動作**。全部差異在 `motion_files/page_diff.txt`。

| 頁 | 誰用 | yuAn_ws | T_Fira_ws | jeff_ws | 我們的機器人 (10-06 備份) |
|---|---|---|---|---|---|
| 1 | sprint 初始化 | test | **stand（不同）** | test（不同） | test（又不同） |
| 30 / 31 / 32 | 避障 爬門 | obstacle_down/down/up | **空的** | 同 yuAn | 同 yuAn ✅ |
| 50 | 射箭 程式一開就做 | 空的 | **ready** | 空的 | 空的 |
| 202 | 射箭 按 START | b_getup_dumb | **ready（拿弓）** | b_getup_dumb | **b_getup_dumb（起身動作！）** |
| 203, 205–209 | 射箭 射 | 空的 | **shoot（不同高度）** | 空的 | 空的 |

- **射箭只能用 T_Fira_ws 的動作檔。** 用我們現在的動作檔跑射箭：按 START 會做「起身」動作 → 很危險。
- 避障的 30/31/32 在我們機器人上跟 jeff_ws 一樣 ✅。
- 第 1 頁（站好的姿勢）每個 workspace 都不一樣，我們機器人的第 1 頁也跟筆電上的不一樣。
- T_Fira_ws 的動作檔跟其他差很多（17–21 魔術、85/86 跌倒起身都是空的），**不能直接整個換掉**。
  要用射箭的話，只能把需要的頁（50, 202, 203, 205–209）**搬到沒有用到的頁碼**，再改程式的頁碼。

### 2. 走路參數 —— 3 個 workspace 都不一樣

（粗體 = 不一樣；`yaw_offset, pelvis_offset, balance_knee_gain, p/i/d_gain` 全部一樣）

| key | 我們機器人 10-06 | jeff_ws | T_Fira_ws | yuAn param | yuAn 500 (para 4) | yuAn back (para 2) |
|---|---|---|---|---|---|---|
| x_offset | -0.015 | -0.015 | -0.008 | -0.03 | -0.02 | -0.02 |
| y_offset | 0.015 | 0.015 | 0.02 | 0.01 | 0.012 | 0.015 |
| z_offset | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 |
| **roll_offset** | **5** | 5 | **-5（正負相反！）** | 3 | 3 | 0 |
| pitch_offset | 0 | 0 | 0 | 3 | 0 | 1 |
| hip_pitch_offset | 10 | 10 | 3 | 8 | 8 | 8 |
| **period_time (ms)** | 800 | 800 | 650 | 700 | **500** | 805 |
| dsp_ratio | 0.5 | 0.5 | 0.3 | 0 | 0 | 0 |
| step_forward_back_ratio | 0.35 | 0.35 | 0.35 | 0.3 | 0.35 | 0.35 |
| foot_height | 0.045 | 0.045 | 0.045 | 0.05 | 0.04 | 0.07 |
| swing_right_left | 0.014 | 0.014 | 0.028 | 0.015 | 0.005 | 0.03 |
| swing_top_down | 0.006 | 0.006 | 0.006 | 0 | 0.003 | 0.01 |
| arm_swing_gain | 0.2 | 0.2 | 0.2 | 0 | 0.5（程式會 ×-1） | 0.5（×-1） |
| balance_hip_roll_gain | 0.5 | 0.5 | 0.5 | 0.1 | 0.1 | 0.1 |
| balance_ankle_roll_gain | 1 | 1 | 0.7 | 0.9 | 0.9 | 0.9 |
| balance_ankle_pitch_gain | 0.9 | 0.9 | 0.9 | 0.7 | 0.7 | 0.7 |

- **jeff_ws 的 param.yaml = 我們現在的**（避障用這個，沒問題）。
- **Sprint 不用 param.yaml 的值走**：它每一步都用 `para='4'`（500.yaml，週期 500 ms，很快）或
  `para='2'`（back.yaml，805 ms）。這些檔在 `yuAn_ws/.../op3_walking_module/config2/`（不是 `config/`），
  路徑**寫死**在 `op3_ros_utils2.py` 裡。`config/` 和 `config2/` 的 `back.yaml` 也不一樣
  （period 800 → 805、swing_right_left 0.035 → 0.030）。

### 3. 函式庫 op3_ros_utils —— 同名但不一樣

| 檔案 | 誰用 | `th`（轉彎） | 其他 |
|---|---|---|---|
| `op3_ros_utils.py`（jeff_ws、T_Fira_ws、yuAn_ws 的 fira_basketball，三個 **完全一樣**） | 避障、射箭 | `th = th - 1`（每一步偷偷 -1 度） | 標準 |
| **我們筆電** `src/fira_basketball/scripts/op3_ros_utils.py` | 我們的舊程式 | **`th = th`（被改過）** | 其他一樣 |
| `yuAn_ws/src/an_pkg/op3_ros_utils2.py` | sprint_canada | 沒有 -1 | `para=` 選 yaml、手臂擺動 ×-1、config2 路徑寫死、6 個 yaml 少一個就當掉 |
| `T_Fira_ws/.../TT_FIRA_2024/scripts/op3_ros_utils2.py` | （這 4 支都沒用） | 沒有 -1 | **跟 yuAn 的同名但完全不同**（整理過的版本，沒有 `para=`） |

- **避障在我們的 workspace 直接跑，會用到我們改過的 `th = th`** → 轉彎跟學長調的時候差 1 度/步，
  機器人會慢慢偏。要照學長的方式就要用 `obstacle_jeff/op3_ros_utils.py`（有 -1）。

### 4. 顏色（HSV）—— 跟場地燈光有關

- 避障 jeff copy 1 跟 benson 原版**只差顏色**：
  - 主程式：紅門 `[0,50,0]→[0,50,37]`、`[170,50,20]→[170,50,126]`
  - 影像程式：黃/藍障礙物 lower `[[20,77,180],[45,119,55]] → [[18,90,110],[100,90,70]]`，
    upper `[[34,255,255],[118,255,255]] → [[32,255,255],[125,255,255]]`
  - 注意：主程式和影像程式的紅色範圍**不一樣**（影像程式沒改）；主程式也有一組 lower/upper 但沒用到。
- 射箭：紅 `[65,228,200]–[180,255,255]`（色相 65–180 非常寬）、黃 `[10,177,219]–[51,249,255]`。
- 這些是**比賽場地的燈光**調出來的，換場地一定要重調。

## 檢查結果（2026-10-13）

| 檢查 | 結果 |
|---|---|
| 複本的程式碼行 = 原檔（拿掉說明和 NOTE 行後逐字比對） | ✅ 7 個 .py 全部一樣 |
| 動作檔複本 md5 = 隨身碟原檔 | ✅ 3 個一樣 |
| 行尾 LF（沒有 CRLF；隨身碟上的原檔是 CRLF） | ✅ |
| Python 2.7 編譯（WSL `python2 -m py_compile`） | ✅ 8 個 .py 都過 |
| pyflakes 找未定義的變數 | ✅ 沒有（只有沒用到的 import/變數） |
| 需要的訊息 `fira_basketball/msg/ThreeDouble.msg` | ✅ 我們 workspace 有 |
| 關節 offset（`op3_tuning_module/data/offset.yaml`） | ✅ 5 個 workspace + 我們機器人**全部一樣** |
| `OP3.robot`、`dxl_init_OP3.yaml` | ✅ 全部一樣 |

**還沒驗證（要在機器人上）**：scipy（射箭要）、`tf`、`cv_bridge` 有沒有裝；機器人上 `/home/robotis/yuAn_ws`、`T_Fira_ws`、
`jeff_ws` 還在不在；這些程式都還**沒有在我們的機器人上跑過**。

## 已知問題（學長的程式本來就有，沒有改）

- **射箭**：程式一開就做第 50 頁（不用按鈕）；`cv2.VideoCapture(0)` 自己開相機，`cv_camera` 不能同時開；
  要螢幕（`cv2.imshow`）；時間到但紅黃中心沒對齊就不射，一直跑下去。
- **避障**：`DOOR_LOCATE` 用到 `target_slop`，如果一開始讀門都失敗就會 `NameError`；影像程式會開視窗，要螢幕。
- **Sprint**：開始時要**站著不動**（前 120 筆 IMU 算陀螺儀偏差）；`WALK_BACKWARDS` 會一直退，要按 MODE 停。

## 隨身碟上還有、但沒複製的（比較新）

| 檔案 | 日期 | 說明 |
|---|---|---|
| `yuAn_ws/src/an_pkg/sprint_canada2.py` | 07-20 16:06 | sprint_canada 的下一版 |
| `T_Fira_ws/.../Archery_canada_clockwise.py`、`Archery_canada_counter_clockwise.py` | 07-17 | 另一套射箭 |
| `T_Fira_ws/.../archery_260425_CLOCKWISE.py`、`..._COUNTER_CLOCKWISE.py` | 07-18 | 較早的射箭 |
| `jeff_ws/.../jeff_obstacle4(copy2).py`、`jeff_obstacle_vision(copy2).py`、`jeff_obstacle4(test).py` | 07-20 | 避障的下一版（copy2 有 40 KB，改很多） |

學長指定的是上面那 4 支，所以先只整理那 4 支。要不要也整理這些，**問學長哪一版才是比賽最後真的用的**。

## 舉重（weightlift_Ray）

2026-10-19 從隨身碟 `E:\Ray_wk` 補上。程式碼和原檔一模一樣（沒有改）。

| 項目 | 原本位置（機器人上） | 這裡 |
|---|---|---|
| 舉重主程式（2026-07-20，最新） | `/home/robotis/Ray_wk/src/fira_basketball/scripts/weightlift/op3_weightlifting_v4.py` | `weightlift_Ray/op3_weightlifting_v4.py` |
| 偵測（OpenVINO，YOLO11n-obb，找 bar_target / white_line） | 同資料夾 `detector.py` + `best_openvino_model/` | `weightlift_Ray/detector.py`, `weightlift_Ray/best_openvino_model/` |
| 函式庫 | `scripts/op3_ros_utils.py` | `weightlift_Ray/op3_ros_utils.py` |
| 動作檔（2026-07-18） | `Ray_wk/src/ROBOTIS-OP3/op3_action_module/data/motion_4095.bin` | `motion_files/motion_4095_Ray_wk.bin` |
| 走路參數 | `Ray_wk/.../op3_walking_module/config/param.yaml` | `walking_params/param_Ray_wk.yaml`（和 jeff_ws 的一樣） |

- 怎麼跑：看 `weightlift_Ray/data.md`。偵測程式在 PC / conda `yolo_ov`（python3 + openvino）上跑，
  用 UDP 5005 把結果送給 `op3_weightlifting_v4.py`（機器人上 python2 + ROS）。
- 用到的動作頁：`1` 初始、`195` 撿槓（30 cd）、`186` 舉起（30 cd）、`185` 結束。
  **頁 185 / 186 / 195 只有 Ray_wk 的動作檔有**，其他三個 workspace 這幾頁是空的。頁 1 四個都不一樣。
- 沒有複製：`jeff_weightlifting_canada1~9`、`jeff_weightlifting_copy25~31`、`weightlifting_fira_seoul2025.py`、
  `weightlifting_test.py`（舊版或測試版）。**要跟學長確認 v4 是不是比賽用的那支。**
