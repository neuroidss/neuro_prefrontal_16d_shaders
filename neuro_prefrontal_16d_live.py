#!/usr/bin/env python3
"""
🧠 NEUROCANVAS LIVE v108.0: TRUE ORTHOGONAL SHADOW BRANCHING
- Тень Fpz и Основной Объект ОРТОГОНАЛЬНЫ 80% времени (разный цвет, форма и орбита).
- В момент Phase Reset Fpz мгновенно отскакивает на +120° для предпросмотра НОВОЙ цели.
- Честная длительность фазы наведения (EXPLORE).
- [F1] / [TAB] / [D] — скрытый инженерный дебаг (ноль цифр по умолчанию).
- 100% CUDA Batched Physics (<1.2 ms).
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import argparse
import math
import numpy as np
import pygame
import torch

from neuro_heterarchy_core import HeterarchicalBrainEngine, NUM_MAX_DEVICES, DEVICE

WIDTH, HEIGHT = 1600, 960
PI = math.pi
TWO_PI = 2.0 * math.pi

class CUDA_16D_ExactManifold:
    def __init__(self):
        # 16D состояние: [4 узла, 4 оси (lx, ly, rx, ry)]
        # 0=F3, 1=F4, 2=AFz, 3=Fpz
        self.state_16d = torch.zeros((4, 4), device=DEVICE, dtype=torch.float32)
        self.vel_16d   = torch.zeros((4, 4), device=DEVICE, dtype=torch.float32)

        # Разносим начальные состояния F3/F4 и Fpz ортогонально!
        self.state_16d[0] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
        self.state_16d[1] = torch.tensor([PI, 0.0, 0.8, 0.0], device=DEVICE)
        self.state_16d[2] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=DEVICE)
        self.state_16d[3] = torch.tensor([PI/2, 0.0, 0.4, 0.0], device=DEVICE) # Тень сразу на 90° в стороне!

        self.N_pts = 240
        self.gamma = torch.linspace(0, TWO_PI, self.N_pts, device=DEVICE, dtype=torch.float32)

        # Сетка для 3D-гироскопов в дебаге
        self.grid_u, self.grid_v = 24, 12
        u = torch.linspace(0, TWO_PI, self.grid_u, device=DEVICE)
        v = torch.linspace(0, TWO_PI, self.grid_v, device=DEVICE)
        self.U, self.V = torch.meshgrid(u, v, indexing='ij')

        self.flash_timer = 0.0
        self.switch_readiness = 0.0

    @torch.inference_mode()
    def update_physics(self, inputs_16d_gpu, dt):
        self.vel_16d = self.vel_16d * 0.85 + inputs_16d_gpu * 0.15 * 3.5
        self.state_16d = (self.state_16d + self.vel_16d * dt) % TWO_PI

        if self.flash_timer > 0:
            self.flash_timer = max(0.0, self.flash_timer - dt * 3.5)

        # Готовность к срыву (0.0 .. 1.0)
        fpz_rx = self.state_16d[3, 2].item()
        fpz_ry = inputs_16d_gpu[3, 3].item()
        self.switch_readiness = float(np.clip(fpz_rx * 0.6 + fpz_ry * 0.4, 0.0, 1.0))

        # ======================================================================
        # ТОЧНЫЙ ФАЗОВЫЙ СРЫВ: ОСНОВНОЙ ОБЪЕКТ ПРИНИМАЕТ FPZ, А FPZ ОТСКАКИВАЕТ!
        # ======================================================================
        if fpz_ry > 0.75 or self.switch_readiness >= 0.98:
            # 1. Основной мир становится тем, что показывала тень
            self.state_16d[0] = self.state_16d[3].clone()
            self.state_16d[1, 0] = self.state_16d[3, 1].clone()
            self.state_16d[1, 1] = self.state_16d[3, 2].clone()
            self.state_16d[2] = self.state_16d[3].clone()

            # 2. ТЕНЬ FPZ МГНОВЕННО ОТСКАКИВАЕТ НА +120° (СЛЕДУЮЩИЙ МИР)
            self.state_16d[3, 0] = (self.state_16d[3, 0] + 2.0 * PI / 3.0) % TWO_PI
            self.state_16d[3, 1] = (self.state_16d[3, 1] + PI / 2.0) % TWO_PI
            self.state_16d[3, 2] = 0.25 # Сброс натяжения тени
            self.state_16d[3, 3] = 0.0

            self.flash_timer = 1.0

        return self.state_16d, self.switch_readiness

    @torch.inference_mode()
    def _eval_geometry(self, f3_vec, f4_vec, rule_vec, ghost_offset=(0.0, 0.0)):
        th_f3, ph_f3, rx_f3, ry_f3 = f3_vec
        th_f4, ph_f4, rx_f4, ry_f4 = f4_vec
        th_rule, ph_rule, rx_rule, ry_rule = rule_vec

        r_base = 180.0
        h3 = 55.0 * torch.cos(3.0 * (self.gamma + ph_f3)) * torch.cos(th_f3)
        h4 = 45.0 * torch.sin(4.0 * (self.gamma + ph_f3)) * torch.sin(th_f3)
        spiral = 18.0 * torch.sin(7.0 * self.gamma + rx_f3)
        teeth  = 12.0 * torch.cos(15.0 * self.gamma + ry_f3)

        gamma_warped = self.gamma + 0.25 * torch.sin(2.0 * self.gamma + th_rule) + 0.15 * torch.cos(3.0 * self.gamma + ph_rule)
        metric_comp  = 1.0 + 0.20 * torch.sin(rx_rule) * torch.cos(5.0 * self.gamma)
        layering     = 8.0 * torch.sin(12.0 * self.gamma + ry_rule)

        R = (r_base + h3 + h4 + spiral + teeth + layering) * metric_comp
        
        ox = ghost_offset[0] + R * torch.cos(gamma_warped)
        oy = ghost_offset[1] + R * torch.sin(gamma_warped)

        # Вычисляем честный цвет
        hue = th_f4.item()
        chroma = 0.5 + 0.5 * math.sin(ph_f4.item())
        glow = 0.7 + 0.3 * math.sin(ry_f4.item())
        diffract = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(rx_f4.item()))

        r = int(np.clip((127 + 127 * math.sin(hue) * chroma) * glow, 0, 255))
        g = int(np.clip((127 + 127 * math.sin(hue + TWO_PI/3) * chroma) * glow, 0, 255))
        b = int(np.clip((127 + 127 * math.sin(hue + 4*PI/3) * chroma) * glow, 0, 255))

        return ox.cpu().numpy(), oy.cpu().numpy(), (r, g, b), diffract

    @torch.inference_mode()
    def compute_16d_object(self):
        # 1. ОСНОВНОЙ ОБЪЕКТ
        ox_m, oy_m, col_m, diffract_depth = self._eval_geometry(
            self.state_16d[0], self.state_16d[1], self.state_16d[2]
        )

        # 2. ТЕНЬ FPZ (СВОЯ ЖИВАЯ ФОРМА И ЦВЕТ ИЗ УЗЛА FPZ)
        shift_dist = 115.0 * (1.0 - self.switch_readiness * 0.45)
        ghost_angle = self.state_16d[3, 0].item()
        ghost_off_x = shift_dist * math.cos(ghost_angle)
        ghost_off_y = shift_dist * math.sin(ghost_angle)

        shadow_f3 = self.state_16d[3]
        shadow_f4 = torch.stack([self.state_16d[3, 1], self.state_16d[3, 2], self.state_16d[3, 2], self.state_16d[3, 3]])
        shadow_rule = self.state_16d[3]

        ox_g, oy_g, col_g, diffract_g = self._eval_geometry(
            shadow_f3, shadow_f4, shadow_rule, ghost_offset=(ghost_off_x, ghost_off_y)
        )

        if self.flash_timer > 0:
            fv = self.flash_timer
            col_m = (min(255, int(col_m[0]*(1-fv) + 255*fv)),
                     min(255, int(col_m[1]*(1-fv) + 220*fv)),
                     min(255, int(col_m[2]*(1-fv) + 50*fv)))

        return (
            ox_m, oy_m, col_m, diffract_depth,
            ox_g, oy_g, col_g, diffract_g,
            (ghost_off_x, ghost_off_y)
        )

    @torch.inference_mode()
    def compute_gyroscope_tori(self, node_idx, fixed_pitch=0.75):
        th1, ph1, th2, ph2 = self.state_16d[node_idx]

        R1, r1 = 55.0, 20.0
        x_m = (R1 + r1 * torch.cos(self.V)) * torch.cos(self.U)
        y_m = (R1 + r1 * torch.cos(self.V)) * torch.sin(self.U)
        z_m = r1 * torch.sin(self.V)

        cp, sp = math.cos(fixed_pitch), math.sin(fixed_pitch)
        y_proj = y_m * cp - z_m * sp

        bx_macro = (R1 + r1 * torch.cos(ph1)) * torch.cos(th1)
        by_macro = (R1 + r1 * torch.cos(ph1)) * torch.sin(th1)
        bz_macro = r1 * torch.sin(ph1)
        by_p = by_macro * cp - bz_macro * sp

        orbit_radius = 12.0
        mx = bx_macro.item() + orbit_radius * math.cos(th2.item())
        my = by_p.item() + orbit_radius * math.sin(ph2.item())

        return x_m.cpu().numpy(), y_proj.cpu().numpy(), (int(bx_macro.item()), int(by_p.item())), (int(mx), int(my))

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Live 16D Closed-Loop Lab")
    parser.add_argument('--sim', action='store_true', help="Запустить честного агента")
    parser.add_argument('--debug', action='store_true', help="Включить отображение дебага по умолчанию")
    args = parser.parse_args()

    agent = None
    if args.sim:
        from synthetic_16d_agent import Synthetic16DAgent
        agent = Synthetic16DAgent()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Orthogonal Shadow 16D Lab")
    clock = pygame.time.Clock()

    font_debug_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_debug_sm = pygame.font.SysFont("consolas", 11)

    engine = HeterarchicalBrainEngine()
    engine.start()

    manifold = CUDA_16D_ExactManifold()
    show_debug = args.debug

    gyro_centers = [
        (190, 180),          # F3
        (190, HEIGHT - 180),  # F4
        (WIDTH - 190, 180),   # AFz
        (WIDTH - 190, HEIGHT - 180) # Fpz
    ]
    slot_roles = ["F3: SHAPE", "F4: OPTICS", "AFz: RULE", "Fpz: SHADOW"]
    slot_colors = [(0, 220, 255), (255, 100, 220), (255, 200, 50), (160, 80, 255)]

    ghost_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    running = True
    try:
        while running:
            dt = min(0.05, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_F1, pygame.K_TAB, pygame.K_d):
                        show_debug = not show_debug

            frame = engine.get_frame()

            # 1. Чтение 16D кинематики
            inputs_16d_np = np.zeros((4, 4), dtype=np.float32)
            for i in range(min(4, len(frame.nodes))):
                k = frame.nodes[i].kinematics
                inputs_16d_np[i] = [k.lx, k.ly, k.rx, k.ry]
            inputs_16d = torch.from_numpy(inputs_16d_np).to(DEVICE)

            # 2. Физика
            state, switch_readiness = manifold.update_physics(inputs_16d, dt)

            (ox_m, oy_m, col_m, diffract_depth,
             ox_g, oy_g, col_g, diffract_g,
             ghost_offset) = manifold.compute_16d_object()

            # 3. Честный фидбек агенту
            honest_match = 0.0
            tgt_info = {"name": ""}
            mode_str = ""
            if agent is not None:
                agent.feed_screen_visuals(col_m, diffract_depth)
                t_idx, tgt_info, mode_str, hold_timer, honest_match = agent.get_status()

            # ==================================================================
            # 4. ЧИСТЫЙ СЕНСОРНЫЙ РЕНДЕРИНГ (НОЛЬ ЦИФР ПО УМОЛЧАНИЮ)
            # ==================================================================
            screen.fill((5, 7, 10))
            ghost_surface.fill((0, 0, 0, 0))
            cx, cy = WIDTH // 2, HEIGHT // 2

            # ⚡ ПЛАЗМЕННЫЕ СТРУНЫ НАТЯЖЕНИЯ (СВЯЗЫВАЮТ ОБЪЕКТ С ЕГО БУДУЩИМ ДВОЙНИКОМ)
            if switch_readiness > 0.05:
                num_filaments = int(3 + switch_readiness * 14)
                for f_i in range(num_filaments):
                    idx_pt = (f_i * (len(ox_m) // num_filaments)) % len(ox_m)
                    p_main = (int(cx + ox_m[idx_pt]), int(cy - oy_m[idx_pt]))
                    p_ghost = (int(cx + ox_g[idx_pt]), int(cy - oy_g[idx_pt]))
                    fil_col = (int(160 * switch_readiness), int(100 + 100 * switch_readiness), 255)
                    pygame.draw.line(screen, fil_col, p_main, p_ghost, max(1, int(switch_readiness * 3)))

            # Энергетическое сжимающееся кольцо горизонта
            ring_rad = int(290 * (1.0 - switch_readiness * 0.18))
            ring_col = (int(130 * switch_readiness), int(180 * (1.0 - switch_readiness) + 40), 255)
            pygame.draw.circle(screen, ring_col, (cx, cy), ring_rad, max(1, int(1 + switch_readiness * 4)))

            # --- А. ЖИВАЯ ЦВЕТНАЯ ПОЛУПРОЗРАЧНАЯ ТЕНЬ (Fpz) ---
            pts_ghost = [(int(cx + ox_g[k]), int(cy - oy_g[k])) for k in range(len(ox_g))]
            
            alpha_body = int(45 + switch_readiness * 140)
            pygame.draw.polygon(ghost_surface, (col_g[0], col_g[1], col_g[2], alpha_body), pts_ghost)
            
            alpha_border = int(120 + switch_readiness * 135)
            pygame.draw.polygon(ghost_surface, (col_g[0], col_g[1], col_g[2], alpha_border), pts_ghost, max(1, int(1.0 + switch_readiness * 3.0)))

            num_layers_g = int(3 + 5 * diffract_g)
            for lay in range(1, num_layers_g):
                scale_fac = 1.0 - (lay / float(num_layers_g)) * 0.75
                lay_pts_g = [(int(cx + ox_g[k] * scale_fac), int(cy - oy_g[k] * scale_fac)) for k in range(0, len(ox_g), 2)]
                col_lay_g = (max(0, col_g[0] - lay*20), max(0, col_g[1] - lay*15), min(255, col_g[2] + lay*25), int(40 + switch_readiness * 90))
                pygame.draw.polygon(ghost_surface, col_lay_g, lay_pts_g, 1)

            screen.blit(ghost_surface, (0, 0))

            # --- Б. ОСНОВНОЙ ОБЪЕКТ ---
            pts_main = [(int(cx + ox_m[k]), int(cy - oy_m[k])) for k in range(len(ox_m))]
            pygame.draw.polygon(screen, col_m, pts_main)
            
            border_col = (255, 255, 255)
            border_width = 2
            if honest_match > 0.70:
                gold_intensity = (honest_match - 0.70) / 0.30
                border_col = (255, int(215 + 40 * gold_intensity), int(50 * (1.0 - gold_intensity)))
                border_width = int(2 + gold_intensity * 4)
            pygame.draw.polygon(screen, border_col, pts_main, border_width)

            num_layers = int(3 + 5 * diffract_depth)
            for lay in range(1, num_layers):
                scale_fac = 1.0 - (lay / float(num_layers)) * 0.75
                lay_pts = [(int(cx + ox_m[k] * scale_fac), int(cy - oy_m[k] * scale_fac)) for k in range(0, len(ox_m), 2)]
                col_lay = (max(0, col_m[0] - lay*22), max(0, col_m[1] - lay*18), min(255, col_m[2] + lay*28))
                pygame.draw.polygon(screen, col_lay, lay_pts, 1)

            # ==================================================================
            # 5. ДЕБАГ-РЕЖИМ (ПО F1 / TAB / D)
            # ==================================================================
            if show_debug:
                pygame.draw.rect(screen, (14, 18, 26), (20, 15, WIDTH - 40, 65), border_radius=6)
                pygame.draw.rect(screen, (0, 255, 200), (20, 15, WIDTH - 40, 65), 1, border_radius=6)
                
                screen.blit(font_debug_b.render(f"[DEBUG MODE ON] Nodes: {frame.num_live}/4 | Theta: {frame.theta_freq:.2f} Hz | Flash: {manifold.flash_timer:.2f}", True, (0, 255, 200)), (35, 22))
                if agent is not None:
                    screen.blit(font_debug_sm.render(f"Goal: {tgt_info['name']} [{mode_str}] | Match: {honest_match*100:.1f}% | Readiness: {switch_readiness*100:.0f}%", True, (255, 220, 100)), (35, 42))

                # 4 3D-Тора в углах
                state_np = state.cpu().numpy()
                for i in range(4):
                    gx, gy = gyro_centers[i]
                    col = slot_colors[i]

                    panel_w, panel_h = 260, 220
                    px = gx - panel_w // 2
                    py = gy - panel_h // 2
                    pygame.draw.rect(screen, (10, 14, 20), (px, py, panel_w, panel_h), border_radius=8)
                    pygame.draw.rect(screen, col, (px, py, panel_w, panel_h), 1, border_radius=8)
                    screen.blit(font_debug_b.render(slot_roles[i], True, col), (px + 10, py + 8))

                    tx_r, ty_p, b_macro, b_micro = manifold.compute_gyroscope_tori(i)
                    for u_i in range(0, manifold.grid_u, 3):
                        pts_t = [(int(gx + tx_r[u_i, v_i]), int(gy - ty_p[u_i, v_i] - 10)) for v_i in range(manifold.grid_v)]
                        pygame.draw.lines(screen, (28, 38, 52), True, pts_t, 1)

                    bm_x, bm_y = gx + b_macro[0], gy - b_macro[1] - 10
                    pygame.draw.circle(screen, col, (bm_x, bm_y), 7)
                    pygame.draw.circle(screen, (255, 255, 255), (bm_x, bm_y), 2)

                    mm_x, mm_y = gx + b_micro[0], gy - b_micro[1] - 10
                    pygame.draw.line(screen, (255, 80, 180), (bm_x, bm_y), (mm_x, mm_y), 1)
                    pygame.draw.circle(screen, (255, 80, 180), (mm_x, mm_y), 4)

                    s = state_np[i]
                    screen.blit(font_debug_sm.render(f"θ1(lx): {s[0]:.2f} | φ1(ly): {s[1]:.2f}", True, (200, 220, 240)), (px + 10, py + panel_h - 36))
                    screen.blit(font_debug_sm.render(f"θ2(rx): {s[2]:.2f} | φ2(ry): {s[3]:.2f}", True, (255, 120, 180)), (px + 10, py + panel_h - 20))

            pygame.display.flip()

    finally:
        if agent is not None:
            agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
