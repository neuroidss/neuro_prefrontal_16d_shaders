#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: COGNITIVE ACTIVE INFERENCE 16D AGENT (v110.0)
- Реализация модели Доу (Daw 2006) и Кёхлина (Koechlin 2003) для префронтальной коры.
- 0% таймеров: переключение правила происходит ТОЛЬКО при обнаружении тупика (stagnation detector).
- 2 Ортогональных Правила (AFz: Прямая Связь 0° vs Инверсия 180°).
- Чистый LSL-вывод (4 узла x 16 каналов, 250 Hz) с фазово-непрерывной бегущей волной.
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 10
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi

# Физические координаты 16 электродов датчика FreeEEG16-alpha2 (в мм)
COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
IS_CORE = np.hypot(COORDS_X, COORDS_Y) < 8.0

# 4 ДЕТЕРМИНИРОВАННЫХ ЦЕЛЕВЫХ МИРА
# Цели 1 и 3 решаются ТОЛЬКО под Правилом А (Симметрия/Синергия, AFz = 0 рад)
# Цели 2 и 4 решаются ТОЛЬКО под Правилом Б (Вихрь/Инверсия, AFz = PI рад)
GOAL_WORLDS = [
    {
        "name": "💎 1. CYAN TRILOBE [Требует: Правило А (0°)]",
        "req_rule_angle": 0.0,
        "target_rgb": np.array([40, 220, 240], dtype=np.float32),
        "f3": [0.0, 0.0, 0.0, 0.0],
        "f4": [math.pi, 0.0, 0.8, 0.0],
        "rule_afz": [0.0, 0.0, 0.0, 0.0]
    },
    {
        "name": "🌪️ 2. RED VORTEX [Требует: Правило Б (180°)]",
        "req_rule_angle": math.pi,
        "target_rgb": np.array([240, 40, 60], dtype=np.float32),
        "f3": [math.pi/2, math.pi/4, 0.8, 0.0],
        "f4": [0.0, math.pi/2, 0.2, 0.8],
        "rule_afz": [math.pi, 0.0, 0.5, 0.0]
    },
    {
        "name": "🧬 3. EMERALD LATTICE [Требует: Правило А (0°)]",
        "req_rule_angle": 0.0,
        "target_rgb": np.array([50, 230, 110], dtype=np.float32),
        "f3": [math.pi, 0.0, 0.0, 0.9],
        "f4": [math.pi/3, 0.0, 0.9, 0.5],
        "rule_afz": [0.0, math.pi/4, 0.8, 0.5]
    },
    {
        "name": "⚡ 4. VIOLET INVERSION [Требует: Правило Б (180°)]",
        "req_rule_angle": math.pi,
        "target_rgb": np.array([180, 40, 255], dtype=np.float32),
        "f3": [math.pi/4, math.pi/2, 0.5, 0.5],
        "f4": [1.5*math.pi, 0.0, 0.0, 1.0],
        "rule_afz": [math.pi, 0.0, 0.0, 0.0]
    }
]

class AgentLSLProcess(mp.Process):
    def __init__(self, shm_dict):
        super().__init__()
        self.daemon = True
        self.shm = shm_dict

    def run(self):
        # 1. Создаем 4 упорядоченных LSL-аутлета
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}')
            outlets.append(StreamOutlet(info))

        print("🤖 [COGNITIVE AGENT] Запущен активный агент префронтального выбора (250 Hz)...")

        start_time = time.time()
        target_idx = 0
        state_mode = "SOLVE_LOCAL" # Режимы: SOLVE_LOCAL -> STAGNATION_BRANCH -> HOLD_SUCCESS -> NEXT_GOAL
        
        stagnation_timer = 0.0
        success_hold_timer = 0.0
        best_match_in_state = 0.0

        current_cmd = np.zeros((4, 4), dtype=np.float32)
        target_cmd = np.zeros((4, 4), dtype=np.float32)

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            cur_goal = GOAL_WORLDS[target_idx]

            # 1. ЗРИТЕЛЬНЫЙ СЕНСИНГ С ЭКРАНА
            screen_r = self.shm['screen_r'].value
            screen_g = self.shm['screen_g'].value
            screen_b = self.shm['screen_b'].value
            screen_diffract = self.shm['screen_diffract'].value

            # Честная оценка зрительного расстояния до цели в пространстве RGB
            cur_rgb = np.array([screen_r, screen_g, screen_b], dtype=np.float32)
            color_dist = float(np.linalg.norm(cur_rgb - cur_goal['target_rgb'])) / 441.67
            color_match = max(0.0, 1.0 - color_dist)
            
            honest_match = float(np.clip(color_match * 0.75 + screen_diffract * 0.25, 0.0, 1.0))
            self.shm['match_score'].value = honest_match

            # ==================================================================
            # 2. КОГНИТИВНЫЙ АЛГОРИТМ ПРИНЯТИЯ РЕШЕНИЙ (ACTIVE INFERENCE)
            # ==================================================================
            if state_mode == "SOLVE_LOCAL":
                # Агент пытается решить задачу в рамках ТЕКУЩЕГО правила AFz
                target_cmd[0] = cur_goal['f3']
                target_cmd[1] = cur_goal['f4']
                
                # Тень Fpz в покое (альтернатива пока не нужна)
                target_cmd[3, 2] = 0.15
                target_cmd[3, 3] = 0.0

                # Отслеживаем прогресс сходимости
                if honest_match > best_match_in_state + 0.02:
                    best_match_in_state = honest_match
                    stagnation_timer = 0.0 # Ошибка падает, продолжаем градиентный спуск!
                else:
                    stagnation_timer += dt # Прогресс остановился!

                # УСПЕХ: Если текущее правило подошло и Match превысил 75%
                if honest_match >= 0.75:
                    state_mode = "HOLD_SUCCESS"
                    success_hold_timer = 0.0
                    print(f"🎯 [AGENT] Цель достигнута под текущим правилом: {cur_goal['name']} (Match: {honest_match*100:.1f}%)")

                # ТУПИК (Prediction Error Stagnation на AFz):
                # Если прошло 1.8 секунды, а Match уперся в потолок <60% -> ТРЕБУЕТСЯ СМЕНА ПРАВИЛА!
                elif stagnation_timer >= 1.8 and honest_match < 0.60:
                    state_mode = "STAGNATION_BRANCH"
                    stagnation_timer = 0.0
                    print(f"⚠️ [STAGNATION DETECTED] Тупик! Текущее правило не позволяет достичь цели (Match: {honest_match*100:.1f}%). Активация ветвления Fpz...")

            elif state_mode == "STAGNATION_BRANCH":
                # Fpz активирует ОРТОГОНАЛЬНОЕ ПРАВИЛО (поворот на 180° / PI)
                # Тень на экране окрашивается в альтернативный цвет и форму!
                target_cmd[3, 0] = cur_goal['req_rule_angle']
                target_cmd[3, 1] = cur_goal['f4'][0] # Цвет целевого мира
                
                # Накачиваем Сагитту Fpz (Тень становится плотной и яркой)
                stagnation_timer += dt
                target_cmd[3, 2] = float(np.clip(stagnation_timer / 1.5, 0.0, 1.0))
                
                # Когда Сагитта натянута -> СРЫВ ФАЗЫ (Phase Reset)
                if stagnation_timer >= 1.5:
                    target_cmd[3, 3] = 1.0 # Импульс срыва!
                    target_cmd[2] = cur_goal['rule_afz'] # AFz переключается на новое правило
                    state_mode = "SOLVE_LOCAL"
                    best_match_in_state = 0.0
                    stagnation_timer = 0.0
                    print(f"💥 [PHASE RESET] Смена парадигмы! Правило переключено на требуемое. Возобновление градиентного спуска...")

            elif state_mode == "HOLD_SUCCESS":
                # Удержание успешного решения
                success_hold_timer += dt
                target_cmd[3, 2] = 0.1 # Тень спокойна
                target_cmd[3, 3] = 0.0

                if success_hold_timer >= 3.5:
                    target_idx = (target_idx + 1) % len(GOAL_WORLDS)
                    state_mode = "SOLVE_LOCAL"
                    best_match_in_state = 0.0
                    stagnation_timer = 0.0
                    print(f"\n➡️ [NEXT GOAL] Переход к следующей задаче: {GOAL_WORLDS[target_idx]['name']}")

            # C1-плавная интерполяция команд (без спектрального разрыва фаз в LSL!)
            current_cmd += (target_cmd - current_cmd) * (dt / 0.30)

            self.shm['target_idx'].value = target_idx
            modes_dict = {"SOLVE_LOCAL": 0, "STAGNATION_BRANCH": 1, "HOLD_SUCCESS": 2}
            self.shm['state_mode'].value = modes_dict.get(state_mode, 0)
            self.shm['hold_timer'].value = float(success_hold_timer if state_mode == "HOLD_SUCCESS" else stagnation_timer)

            # ==================================================================
            # 3. СИНТЕЗ 16-КАНАЛЬНОЙ БИОФИЗИКИ ЭЭГ
            # ==================================================================
            theta_phase = 2.0 * math.pi * 6.0 * t_vec
            gamma_phase = 2.0 * math.pi * 55.0 * t_vec
            env_nucleus = (np.clip(np.cos(theta_phase), 0, 1) ** 2)
            noise = np.random.normal(0, 0.015, (NUM_CHANNELS, len(t_vec)))

            for node_i in range(NUM_DEVICES):
                cmd_lx, cmd_ly, cmd_rx, cmd_ry = current_cmd[node_i]

                # Формируем волновой вектор kx, ky по координатам электродов
                spatial_phase = (COORDS_X * (cmd_lx * 0.35) + COORDS_Y * (cmd_ly * 0.35))[:, None]
                curl_mod = np.where(IS_CORE, -cmd_rx * 0.7, cmd_rx * 0.7)[:, None]
                pwr_mod = (1.0 + cmd_ry * 0.4)

                raw_sig = 10.0 * np.sin(theta_phase + curl_mod * 0.15) + \
                          pwr_mod * 4.5 * env_nucleus * np.sin(gamma_phase + spatial_phase + curl_mod) + noise

                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)

class Synthetic16DAgent:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value('b', True),
            'match_score': mp.Value('d', 0.0),
            'target_idx': mp.Value('i', 0),
            'state_mode': mp.Value('i', 0),
            'hold_timer': mp.Value('d', 0.0),
            'screen_r': mp.Value('d', 127.0),
            'screen_g': mp.Value('d', 127.0),
            'screen_b': mp.Value('d', 127.0),
            'screen_diffract': mp.Value('d', 0.5)
        }
        self.process = AgentLSLProcess(self.shm)
        self.process.start()

    def feed_screen_visuals(self, rgb, diffract):
        self.shm['screen_r'].value = float(rgb[0])
        self.shm['screen_g'].value = float(rgb[1])
        self.shm['screen_b'].value = float(rgb[2])
        self.shm['screen_diffract'].value = float(diffract)

    def get_status(self):
        t_idx = self.shm['target_idx'].value
        modes = ["SOLVE (Локальный поиск)", "BRANCH (Тупик -> Смена правила)", "HOLD (Успех)"]
        timer = self.shm['hold_timer'].value
        match_score = self.shm['match_score'].value
        return t_idx, GOAL_WORLDS[t_idx], modes[self.shm['state_mode'].value], timer, match_score

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
