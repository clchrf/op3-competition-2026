# competition_2026

> 2026-10-13 從隨身碟（E:）整理。學長說這是**去年比賽用的程式**，分別在**不同的 workspace**，
> 而且**同名的東西在不同 workspace 裡參數不一樣**（差異已用 diff 比過，見下面）。
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

- 避障用的是 `jeff_obstacle4 (copy 1).py` + `jeff_obstacle_vision (copy 1).py` 這一組：主程式 publish
  `obstacle_head_angle`、subscribe `obstacle_vision_data`（`ThreeDouble`），影像程式剛好相反，topic 對得上。
  jeff_ws 裡另一支 `jeff_obstacle.py`（159 行）是單一程式版：自己 subscribe 相機和 IMU，沒有這兩個 topic，所以不是這一組，沒有複製。
- 檔名裡的空白和括號拿掉了（`rosrun` 不喜歡空白）。

每個資料夾都放了**那支程式需要的東西**（它 import 的那個版本的函式庫、它用的走路參數檔）。

```
competition_2026/
  sprint_yuAn/      sprint_canada.py + op3_ros_utils2.py (yuAn 版) + walking_config2/*.yaml (7 個)
  archery_TFira/    archery_260718_COUNTER_CLOCKWISE.py + op3_ros_utils.py (T_Fira 版)
  obstacle_jeff/    jeff_obstacle4_copy1.py + jeff_obstacle_vision_copy1.py + op3_ros_utils.py (jeff 版)
  weightlift_Ray/   舉重 op3_weightlifting_v4.py + detector.py + YOLO 模型 + op3_ros_utils.py (Ray_wk 版)
  base_workspace/   各 workspace 不一樣的基底檔：dxl_init_OP3.yaml、fira_basketball 的 msg/CMakeLists/package.xml/param.yaml
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
| `op3_ros_utils.py`（jeff_ws、T_Fira_ws、yuAn_ws 的 fira_basketball，三個 **完全一樣**，md5 相同） | 避障、射箭 | `th = th - 1`（每一步 -1 度） | 標準 |
| `Ray_wk/src/fira_basketball/scripts/op3_ros_utils.py` | 舉重 v4 | `th = th - 1`（一樣） | 多了 `loadWalkingParams()`，路徑**寫死** `/home/robotis/Ray_wk/src/fira_basketball/config`，開機時 `base_walking_params` 從那裡的 `param.yaml` 讀 |
| **我們筆電** `src/fira_basketball/scripts/op3_ros_utils.py` | 我們的舊程式 | **`th = th`（被改過）** | 其他一樣 |
| `yuAn_ws/src/an_pkg/op3_ros_utils2.py` | sprint_canada | 沒有 -1 | `para=` 選 yaml、手臂擺動 ×-1、config2 路徑寫死、6 個 yaml 少一個就當掉 |
| `T_Fira_ws/.../TT_FIRA_2024/scripts/op3_ros_utils2.py` | （這 4 支都沒用） | 沒有 -1 | **跟 yuAn 的同名但完全不同**（整理過的版本，沒有 `para=`） |

- **避障在我們的 workspace 直接跑，會用到我們改過的 `th = th`** → 每一步的轉彎量比學長的多 1 度（他們是 `th - 1`）。
  要照學長的方式就要用 `obstacle_jeff/op3_ros_utils.py`（有 -1）。

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
| `OP3.robot` | ✅ yuAn / T_Fira / jeff / Ray_wk 全部一樣（夾爪 ID 21、22，XM430-W350，P2.0） |
| `dxl_init_OP3.yaml` | yuAn / T_Fira / jeff 一樣；**Ray_wk 不同**（夾爪初始設定，見最後一節） |

**程式需要的套件**（從 import 看）：射箭 `scipy.optimize`；避障 `tf`（`euler_from_quaternion`）、`matplotlib`、
`fira_basketball.msg.ThreeDouble`；舉重偵測 `openvino`（python3）。這些程式都還**沒有在我們的機器人上跑過**。

## 已知問題（學長的程式本來就有，沒有改）

- **射箭**：程式一開就做第 50 頁（不用按鈕）；`cv2.VideoCapture(0)` 自己開相機，`cv_camera` 不能同時開；
  要螢幕（`cv2.imshow`）；時間到但紅黃中心沒對齊就不射，一直跑下去。
- **避障**：`DOOR_LOCATE` 用到 `target_slop`，如果一開始讀門都失敗就會 `NameError`；影像程式會開視窗，要螢幕。
- **Sprint**：開始時要**站著不動**（前 120 筆 IMU 算陀螺儀偏差）；`WALK_BACKWARDS` 會一直退，要按 MODE 停。

## 隨身碟上還有、但沒複製的

這 4 支（sprint_canada、archery_260718_COUNTER_CLOCKWISE、jeff_obstacle4 / jeff_obstacle_vision 的 copy 1）是指定的，只整理這些。
隨身碟上還有下面這些，內容差異是用 diff 看過的：

| 檔案 | 修改時間 | 和放進來的那份差在哪 |
|---|---|---|
| `yuAn_ws/src/an_pkg/sprint_canada2.py` | 07-20 16:06 | 同一個程式的下一版：`slow_time` 2→2.4、`stop_time` 1→2、`preback_time` 2→2.5；停下前多兩段減速（`x=20`→`x=10`→`x=3`），並用 `para='6'`（`slow.yaml`）；後退多兩段（`x=-10`→`-15`→`-20`），最後 `x=-25`→`-30` |
| `jeff_ws/.../jeff_obstacle4(copy2).py` | 07-20 12:08 | 1160 行（copy 1 是 594 行）；`jeff_obstacle_vision(copy2).py` 多 publish `obstacle_closest_z`（`Float32`） |
| `jeff_ws/.../jeff_obstacle4(test).py` | 07-20 12:07 | 621 行，topic 跟 copy 1 一樣 |
| `T_Fira_ws/.../Archery_canada_clockwise.py`、`Archery_canada_counter_clockwise.py` | 07-17 | 另一套射箭（約 18 KB / 20 KB，比 260718 版的 11 KB 大） |
| `T_Fira_ws/.../archery_260425_CLOCKWISE.py`、`..._COUNTER_CLOCKWISE.py` | 07-18 | 同系列檔名的射箭程式（順時針 / 逆時針） |

要用這些，從隨身碟的原路徑拿。

## 舉重（weightlift_Ray）

2026-10-19 從隨身碟 `E:\Ray_wk` 補上。程式碼和原檔一模一樣（沒有改）。

| 項目 | 原本位置（機器人上） | 這裡 |
|---|---|---|
| 舉重主程式（2026-07-20，最新） | `/home/robotis/Ray_wk/src/fira_basketball/scripts/weightlift/op3_weightlifting_v4.py` | `weightlift_Ray/op3_weightlifting_v4.py` |
| 偵測（OpenVINO，YOLO11n-obb，找 bar_target / white_line） | 同資料夾 `detector.py` + `best_openvino_model/` | `weightlift_Ray/detector.py`, `weightlift_Ray/best_openvino_model/` |
| 函式庫 | `scripts/op3_ros_utils.py` | `weightlift_Ray/op3_ros_utils.py` |
| 動作檔（2026-07-18） | `Ray_wk/src/ROBOTIS-OP3/op3_action_module/data/motion_4095.bin` | `motion_files/motion_4095_Ray_wk.bin` |
| 走路參數 | `Ray_wk/.../op3_walking_module/config/param.yaml` | `walking_params/param_Ray_wk.yaml`（和 jeff_ws 的一樣） |

- 怎麼跑：`weightlift_Ray/data.md` 的指令（路徑都是 `/home/robotis/Ray_wk/...`，**兩支都在機器人上跑**）：
  1. 偵測：`conda activate yolo_ov` → `unset PYTHONPATH` → `python3 weightlift/detector.py`
     （`detector.py` 自己用 `cv2.VideoCapture(0)` 開相機、OpenVINO 跑 `best_openvino_model`，
     把結果用 UDP 送到 `127.0.0.1:5005`，瀏覽器預覽在 `http://127.0.0.1:8080/`）。
  2. 主程式：`source /opt/ros/kinetic/setup.bash`、`source ~/Ray_wk/devel/setup.bash`、
     `export PYTHONPATH=$PYTHONPATH:/home/robotis/Ray_wk/src/fira_basketball/scripts`，再 `python op3_weightlifting_v4.py`
     （python2 + ROS，在 `0.0.0.0:5005` 收偵測結果，超過 0.3 秒沒更新就當作沒看到）。
- 模型 `best_openvino_model`：YOLO11n-obb，輸入 320×320，兩個類別 `bar_target`、`white_line`（`metadata.yaml`）。
- 用到的動作頁：`1` 初始、`195` 撿槓、`186` 舉起、`185` 結束；程式註解標 `#30 cd`（槓鈴上的 CD 片數），
  `196` 是 `#60 cd` 的撿槓頁（在程式裡被註解掉，用 195）。
  **頁 185 / 186 / 195 / 196 只有 Ray_wk 的動作檔有**，其他三個 workspace 這幾頁是空的。頁 1 四個都不一樣。
- 為什麼是 v4：`data.md` 的啟動指令就是 `op3_weightlifting_v4.py`；它也是 `weightlift/` 資料夾裡最後修改的檔案（07-20 19:21），
  動作檔 `motion_4095.bin` 修改時間是 07-18 08:23。
- 沒有複製的舊版：`jeff_weightlifting_canada1~7`（05-22～05-31，用黃色 HSV 找槓，動作頁也是 1 / 195 / 186 / 185）、
  `weightlift/weightlifting_test.py`（07-01，頁號一樣，也用 socket 收偵測）、`jeff_weightlifting_copy25~31`、
  `weightlifting_fira_seoul2025.py`（動作頁 1 / 178 / 192 / 191）。

## 各 workspace 基底有哪裡不一樣（2026-10-19 用 diff 比過）

比對方式：Ray_wk 對 yuAn_ws / T_Fira_ws / jeff_ws，比 `ROBOTIS-*`、`DynamixelSDK` 等 ROBOTIS 套件和 `fira_basketball`
（`scripts/` 以外）。**只有下面這些檔案不一樣**，其他（OP3.robot、offset.yaml、op3_gui_demo、各模組原始碼、launch）四個都一樣。

| 檔案 | yuAn_ws | T_Fira_ws | jeff_ws | Ray_wk | 這裡 |
|---|---|---|---|---|---|
| `op3_action_module/data/motion_4095.bin` | 不同 | 不同 | 不同 | 不同 | `motion_files/` |
| `op3_walking_module/config/param.yaml` | 不同 | 不同 | = Ray | = jeff | `walking_params/` |
| `op3_walking_module/config/{500,back,back2,front,stop}.yaml` + `config2/` | 只有 yuAn 有 | - | - | - | `sprint_yuAn/walking_config2/` |
| `op3_manager/config/dxl_init_OP3.yaml` | A | A | A | B | `base_workspace/dxl_init/` |
| `op3_ros_utils*.py`（程式 import 的函式庫） | 各自版本 | 各自版本 | 各自版本 | 各自版本 | 各程式資料夾 |
| `fira_basketball/msg/ThreeDouble.msg` + `CMakeLists.txt`（`add_message_files`）+ `package.xml` | 有 | - | 有（= yuAn） | **沒有 msg**（CMake 沒開） | `base_workspace/fira_basketball_pkg/` |
| `fira_basketball/config/param.yaml` | = jeff（period_time 600） | - | = yuAn（600） | 不同（period_time 800，其他一樣） | `base_workspace/fira_basketball_config/` |

- `dxl_init_OP3.yaml`：A = yuAn / T_Fira / jeff（三個完全一樣），有 `l_gripper`、`r_gripper` 的初始設定
  （`operating_mode: 3`、`position_p_gain: 800`、`shutdown: 56` …）；
  B = Ray_wk，這兩段換成註解掉的 `l_grip` / `r_grip`，也就是**沒有** `l_gripper` / `r_gripper` 的初始設定。
  四個 workspace 的 `OP3.robot` 完全一樣（夾爪 ID 21 / 22，XM430-W350），所以馬達型號沒有差，差的只有開機時寫進夾爪的設定。
- 避障（`obstacle_jeff`）要 `from fira_basketball.msg import ThreeDouble`：用 jeff 的 `fira_basketball_pkg/`，
  `catkin_make` 之後才有這個訊息。Ray_wk 的 `fira_basketball` 沒有 `msg/` 資料夾，`CMakeLists.txt` 裡
  `add_message_files` 是註解掉的，所以沒有這個訊息。
- 走路 `param.yaml` 重點差異（yuAn / T_Fira / Ray(=jeff)）：period_time 700 / 650 / 800，
  hip_pitch_offset 8 / 3 / 10，x_offset -0.030 / -0.008 / -0.015，foot_height 0.050 / 0.045 / 0.045。
- `an_pkg` 只有 yuAn_ws 有，`TT_FIRA_2024` 只有 T_Fira_ws 有（它的 `config/` 和 jeff_ws 的 `fira_basketball/config/` 一樣）。
  `scripts/` 裡的其他程式沒有比，只放了指定的那幾支。
