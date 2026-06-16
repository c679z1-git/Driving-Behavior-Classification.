"""
Usage:
  python data_collection/carla_recorder.py --behavior normal
  python data_collection/carla_recorder.py --behavior swerving
  python data_collection/carla_recorder.py --behavior tailgating
"""

from __future__ import print_function

import argparse
import glob
import math
import os
import queue
import random
import threading

import carla

try:
    import numpy as np
except ImportError:
    raise RuntimeError("numpy is required: pip install numpy")


def reorganize_clips(bd):
    pat = os.path.join(bd, 'clip_*')
    ex  = sorted(
        d for d in glob.glob(pat)
        if os.path.isdir(d) and os.path.basename(d).startswith('clip_')
    )
    if not ex:
        return 0
    tmp = []
    for src in ex:
        t = src + '__reorg_tmp__'
        os.rename(src, t)
        tmp.append(t)
    for ni, t in enumerate(tmp):
        dst = os.path.join(bd, f'clip_{ni:03d}')
        os.rename(t, dst)
    return len(tmp)


def _make_writer():
    try:
        import cv2
        def write(raw, w, h, fp):
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
            cv2.imwrite(fp, arr[:, :, :3])
        return write
    except ImportError:
        pass
    try:
        from PIL import Image as PILImage
        def write(raw, w, h, fp):
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
            PILImage.fromarray(arr[:, :, [2, 1, 0]]).save(fp)
        return write
    except ImportError:
        pass
    raise RuntimeError("Install opencv-python or Pillow.")


def _disk_writer_thread(sq, se, wf):
    while not se.is_set():
        try:
            raw, w, h, fp = sq.get(timeout=0.5)
            wf(raw, w, h, fp)
            sq.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n[Writer] Error: {e}")


def flush_queue(sq, se, wf):
    if sq.qsize() == 0:
        return
    print(f"Flushing {sq.qsize()} frames to disk...")
    se2  = threading.Event()
    drn  = threading.Thread(target=_disk_writer_thread, args=(sq, se2, wf), daemon=True)
    drn.start()
    sq.join()
    se2.set()
    drn.join(timeout=60)


def thin_dir(sd2, tfps, sfps=60):
    ff = sorted(glob.glob(os.path.join(sd2, 'frame_*.png')))
    if not ff:
        return 0
    ke = max(1, round(sfps / tfps))
    if ke == 1:
        return len(ff)
    kp = set(ff[i] for i in range(0, len(ff), ke))
    for f in ff:
        if f not in kp:
            os.remove(f)
    sv  = sorted(glob.glob(os.path.join(sd2, 'frame_*.png')))
    tmp = [f + '.tmp' for f in sv]
    for old, t in zip(sv, tmp):
        os.rename(old, t)
    for i, t in enumerate(tmp):
        os.rename(t, os.path.join(sd2, 'frame_%06d.png' % i))
    return len(sv)


def spawn_traffic(cl, wd, tm, nv=80, nw=20):
    bpl   = wd.get_blueprint_library()
    tmp   = tm.get_port()
    cbps  = [bp for bp in bpl.filter('vehicle.*')
             if int(bp.get_attribute('number_of_wheels')) == 4]

    sps = wd.get_map().get_spawn_points()
    random.shuffle(sps)

    cmds = []
    for sp in sps[:nv]:
        bp = random.choice(cbps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        if bp.has_attribute('driver_id'):
            bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        cmds.append(carla.command.SpawnActor(bp, sp)
                    .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tmp)))

    res  = cl.apply_batch_sync(cmds, True)
    nvs  = [wd.get_actor(r.actor_id) for r in res
            if not r.error and wd.get_actor(r.actor_id)]
    print(f"[Traffic] {len(nvs)}/{nv} vehicles spawned.")

    wbps = bpl.filter('walker.pedestrian.*')
    cbp  = bpl.find('controller.ai.walker')

    wc = []
    for _ in range(nw):
        loc = wd.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = random.choice(wbps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        wc.append(carla.command.SpawnActor(bp, carla.Transform(loc)))

    wr   = cl.apply_batch_sync(wc, True)
    wlks = [wd.get_actor(r.actor_id) for r in wr
            if not r.error and wd.get_actor(r.actor_id)]
    cc   = [carla.command.SpawnActor(cbp, carla.Transform(), w.id) for w in wlks]
    cr2  = cl.apply_batch_sync(cc, True)
    cts  = [wd.get_actor(r.actor_id) for r in cr2
            if not r.error and wd.get_actor(r.actor_id)]

    wd.tick()
    for ct in cts:
        ct.start()
        dst = wd.get_random_location_from_navigation()
        if dst:
            ct.go_to_location(dst)
        ct.set_max_speed(1.0 + random.random())

    print(f"[Traffic] {len(wlks)} pedestrians spawned.")
    return nvs, wlks, cts


def speed(ac):
    v = ac.get_velocity()
    return math.sqrt(v.x**2 + v.y**2)


def pick_moving_npc(nvs, excl=None, nm=5):
    excl = excl or set()
    mv = [(speed(v), v) for v in nvs
          if v.is_alive and v.id not in excl and speed(v) > 1.0]
    if len(mv) < nm:
        return None
    mv.sort(key=lambda x: x[0])
    return mv[len(mv) // 2][1]


def spawn_ego(cl, wd, bpl, col='255,0,0'):
    ebps = list(bpl.filter('vehicle.tesla.model3')) or \
           [bp for bp in bpl.filter('vehicle.*')
            if int(bp.get_attribute('number_of_wheels')) == 4]
    ebp = random.choice(ebps)
    ebp.set_attribute('role_name', 'hero')
    if ebp.has_attribute('color'):
        ebp.set_attribute('color', col)
    sps = wd.get_map().get_spawn_points()
    random.shuffle(sps)
    for sp in sps:
        res = cl.apply_batch_sync([carla.command.SpawnActor(ebp, sp)], True)
        if res and not res[0].error:
            ac = wd.get_actor(res[0].actor_id)
            if ac:
                return ac
    return None


def destroy_actor(ac):
    if ac and ac.is_alive:
        try:
            ac.destroy()
        except Exception:
            pass


TG  = 2.5
CG  = 1.2
LAM = 6.0
VHL = 2.5
MSK = 80.0


def _angle_to_aim(etf, al):
    el  = etf.location
    ey  = math.radians(etf.rotation.yaw)
    dx  = al.x - el.x
    dy  = al.y - el.y
    dst = math.sqrt(dx * dx + dy * dy)
    if dst < 0.001:
        return 0.0, 0.0
    ay  = math.atan2(dy, dx)
    alp = ay - ey
    alp = (alp + math.pi) % (2 * math.pi) - math.pi
    return alp, dst


def _pure_pursuit_steer(etf, al, es, wb=2.87, ms=0.75):
    alp, ld = _angle_to_aim(etf, al)
    if ld < 0.1:
        return 0.0
    sr  = math.atan2(2.0 * wb * math.sin(alp), ld)
    sf  = min(1.0, max(0.3, 14.0 / max(es, 1.0)))
    return float(max(-ms, min(ms, sr * sf)))


class TailgatingController:
    KP_SPD = 0.18
    KI_SPD = 0.02
    KP_GAP = 0.06
    MAX_THROTTLE = 0.80
    MAX_BRAKE    = 0.80

    def __init__(self, cm):
        self._map = cm
        self._si  = 0.0
        self._gh  = []

    def reset(self):
        self._si = 0.0
        self._gh = []

    def _smooth_gap(self, rg, wn=5):
        self._gh.append(rg)
        if len(self._gh) > wn:
            self._gh.pop(0)
        return sum(self._gh) / len(self._gh)

    def _aim_point(self, tgt):
        twp = self._map.get_waypoint(
            tgt.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if twp is None:
            return tgt.get_location()
        ac2 = 0.0
        wp  = twp
        stp = 2.0
        while ac2 < LAM:
            nx = wp.next(stp)
            if not nx:
                break
            wp  = nx[0]
            ac2 += stp
        return wp.transform.location

    def run_step(self, ego, tgt, dt):
        if not ego.is_alive or not tgt.is_alive:
            return carla.VehicleControl(brake=1.0, hand_brake=False)
        etf  = ego.get_transform()
        es   = speed(ego)
        ts   = speed(tgt)
        rd   = ego.get_location().distance(tgt.get_location())
        rg   = max(0.0, rd - 2 * VHL)
        gap  = self._smooth_gap(rg)
        if gap < CG:
            eb  = min(1.0, 0.5 + (CG - gap) / CG * 0.5)
            aim = self._aim_point(tgt)
            st  = _pure_pursuit_steer(etf, aim, es)
            return carla.VehicleControl(throttle=0.0, brake=float(eb),
                                        steer=st, hand_brake=False)
        se2 = ts - es
        if abs(gap - TG) < 8.0:
            self._si += se2 * dt
            self._si  = max(-5.0, min(5.0, self._si))
        ge  = gap - TG
        ld2 = (self.KP_SPD * se2 + self.KI_SPD * self._si + self.KP_GAP * ge)
        if es * 3.6 >= MSK and ld2 > 0:
            ld2 = 0.0
        if ld2 >= 0:
            th = float(min(self.MAX_THROTTLE, ld2))
            br = 0.0
        else:
            th  = 0.0
            bsc = 1.0 if gap < TG + 2.0 else 0.4
            br  = float(min(self.MAX_BRAKE, abs(ld2) * bsc))
        aim = self._aim_point(tgt)
        st  = _pure_pursuit_steer(etf, aim, es)
        ct  = carla.VehicleControl()
        ct.throttle   = th
        ct.brake      = br
        ct.steer      = st
        ct.hand_brake = False
        return ct


def run_clip_tailgating(wd, cl, bpl, tm, nvs):
    cm  = wd.get_map()
    tgt = None
    for _ in range(20 * 60):
        cds = [(speed(v), v) for v in nvs if v.is_alive and speed(v) > 2.0]
        if len(cds) >= 3:
            cds.sort(key=lambda x: x[0])
            tgt = cds[len(cds) // 2][1]
            break
        wd.tick()
    if tgt is None:
        raise RuntimeError("No moving NPC available for tailgating target.")
    twp = cm.get_waypoint(tgt.get_location(), project_to_road=True,
                          lane_type=carla.LaneType.Driving)
    if twp is None:
        raise RuntimeError("Target has no road waypoint.")
    sbm = max(8.0, TG + 2 * VHL + 1.0)
    swp = twp
    tv  = 0.0
    stp = 2.0
    while tv < sbm:
        pv = swp.previous(stp)
        if not pv:
            break
        swp  = pv[0]
        tv  += stp
    stf           = swp.transform
    stf.location.z += 0.35
    ebps = list(bpl.filter('vehicle.tesla.model3')) or \
           [bp for bp in bpl.filter('vehicle.*')
            if int(bp.get_attribute('number_of_wheels')) == 4]
    ebp = random.choice(ebps)
    ebp.set_attribute('role_name', 'hero')
    if ebp.has_attribute('color'):
        ebp.set_attribute('color', '255,0,0')
    ego = None
    res = cl.apply_batch_sync([carla.command.SpawnActor(ebp, stf)], True)
    if res and not res[0].error:
        ego = wd.get_actor(res[0].actor_id)
    if ego is None:
        rl  = stf.location
        sps = cm.get_spawn_points()
        sps.sort(key=lambda sp: sp.location.distance(rl))
        for sp in sps[:15]:
            res = cl.apply_batch_sync([carla.command.SpawnActor(ebp, sp)], True)
            if res and not res[0].error:
                ego = wd.get_actor(res[0].actor_id)
                if ego:
                    break
    if ego is None:
        raise RuntimeError("Could not spawn tailgating ego.")
    ego.set_autopilot(False, tm.get_port())
    hld = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=True)
    for _ in range(60):
        ego.apply_control(hld)
        wd.tick()
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5,
                                           steer=0.0, hand_brake=False))
    wd.tick()
    ct  = TailgatingController(cm)
    id2 = ego.get_location().distance(tgt.get_location())
    print(f"  [Tailgating] ego={ego.id} -> target={tgt.id} "
          f"tgt_spd={speed(tgt)*3.6:.1f} km/h  init_dist={id2:.1f} m")
    return ego, ct, tgt


def tailgating_control(ego, ct, tgt, dt):
    if not ego.is_alive or not tgt.is_alive:
        return False
    ego.apply_control(ct.run_step(ego, tgt, dt))
    return True


def run_clip_swerving(wd, cl, bpl, tm):
    ego = spawn_ego(cl, wd, bpl, col='0,0,255')
    if ego is None:
        raise RuntimeError("Could not spawn swerving ego.")
    ego.set_autopilot(False, tm.get_port())
    for _ in range(60):
        wd.tick()
    print(f"  [Swerving] ego={ego.id} spawned.")
    return ego


def swerve_control(ego, st2, tsk=45.0, dt=1.0/60.0):
    st2['t'] += dt
    if st2['t'] >= st2['period']:
        st2['t']         = 0.0
        st2['period']    = random.uniform(2.0, 5.0)
        st2['amplitude'] = random.uniform(0.2, 0.5)
    sr  = st2['amplitude'] * math.sin((st2['t'] / st2['period']) * 2 * math.pi)
    tms = tsk / 3.6
    sv  = speed(ego)
    er  = tms - sv
    ct  = carla.VehicleControl()
    ct.steer      = float(sr)
    ct.hand_brake = False
    if er > 0.5:
        ct.throttle = min(0.8, 0.3 + er * 0.05)
        ct.brake    = 0.0
    elif er < -1.0:
        ct.throttle = 0.0
        ct.brake    = min(0.5, abs(er) * 0.05)
    else:
        ct.throttle, ct.brake = 0.15, 0.0
    ego.apply_control(ct)
    return st2


def record(args):
    bh  = args.behavior
    W   = args.width
    H   = args.height
    nc  = args.clips
    cs  = args.clip_duration
    cfn = int(cs * 60)
    bd  = os.path.join("dataset", bh)
    os.makedirs(bd, exist_ok=True)

    sci = reorganize_clips(bd)

    cl  = carla.Client(args.host, args.port)
    cl.set_timeout(30.0)
    wd  = cl.get_world()

    os2 = wd.get_settings()
    s2  = wd.get_settings()
    s2.synchronous_mode    = True
    s2.fixed_delta_seconds = 1.0 / 60.0
    wd.apply_settings(s2)

    if args.no_render:
        s2.no_rendering_mode = True
        wd.apply_settings(s2)

    tm = cl.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_random_device_seed(42)

    for _ in range(10):
        wd.tick()

    wf  = _make_writer()
    sq  = queue.Queue(maxsize=600)
    se  = threading.Event()
    wrt = threading.Thread(target=_disk_writer_thread, args=(sq, se, wf), daemon=True)
    wrt.start()

    nvs, nwl, nct = [], [], []
    cam = None
    ego = None
    cf  = [0]
    cd2 = ['']

    def on_image(image):
        fp  = os.path.join(cd2[0], f'frame_{cf[0]:06d}.png')
        raw = bytes(image.raw_data)
        try:
            sq.put_nowait((raw, image.width, image.height, fp))
            cf[0] += 1
        except queue.Full:
            pass

    bpl    = wd.get_blueprint_library()
    cbp    = bpl.find('sensor.camera.rgb')
    cbp.set_attribute('image_size_x', str(W))
    cbp.set_attribute('image_size_y', str(H))
    cbp.set_attribute('fov', '90')
    cbp.set_attribute('gamma', '2.2')
    ctf    = carla.Transform(carla.Location(x=1.2, y=0.0, z=1.3))

    def attach_cam(ta):
        cam2 = wd.spawn_actor(cbp, ctf, attach_to=ta,
                              attachment_type=carla.AttachmentType.Rigid)
        cam2.listen(on_image)
        return cam2

    ts  = 0
    dt  = 1.0 / 60.0
    uni = set()

    print(f"\n[Recorder] Behavior  : {bh.upper()}")
    print(f"[Recorder] Clips     : {nc} x {cs}s")
    print(f"[Recorder] New clips : clip_{sci:03d} -> clip_{sci + nc - 1:03d}")
    if not args.no_thin:
        print(f"[Recorder] Auto-thin : {args.target_fps}fps after each clip")
    print()

    try:
        nvs, nwl, nct = spawn_traffic(cl, wd, tm, args.num_vehicles, args.num_walkers)
        for _ in range(15 * 60):
            wd.tick()
            if pick_moving_npc(nvs) is not None:
                break

        ss = {'t': 0.0, 'period': random.uniform(2.5, 4.5),
              'amplitude': random.uniform(0.25, 0.45)}

        for si in range(nc):
            ci  = sci + si
            cld = os.path.join(bd, f'clip_{ci:03d}')
            os.makedirs(cld, exist_ok=True)
            cd2[0] = cld
            cf[0]  = 0
            hv     = None
            tv2    = None
            tgc    = None

            print(f"[Clip {ci:03d}] Starting ({bh})...")

            if bh == 'normal':
                hv = pick_moving_npc(nvs, excl=uni)
                if hv is None:
                    uni.clear()
                    hv = pick_moving_npc(nvs)
                if hv is None:
                    print(f"[Clip {ci:03d}] No moving NPC — skipping.")
                    continue
                uni.add(hv.id)
                print(f"  [Normal] Riding NPC id={hv.id} ({hv.type_id})")

            elif bh == 'tailgating':
                ok2 = False
                for att in range(3):
                    try:
                        ego, tgc, tv2 = run_clip_tailgating(wd, cl, bpl, tm, nvs)
                        hv   = ego
                        ok2  = True
                        break
                    except RuntimeError as e:
                        print(f"  [Tailgating] Setup attempt {att+1}/3 failed: {e} — retrying...")
                        for _ in range(5 * 60):
                            wd.tick()
                if not ok2:
                    print(f"[Clip {ci:03d}] Could not set up tailgating — skipping.")
                    import shutil; shutil.rmtree(cld, ignore_errors=True)
                    continue

            elif bh == 'swerving':
                ego = run_clip_swerving(wd, cl, bpl, tm)
                hv  = ego
                ss  = {'t': 0.0, 'period': random.uniform(2.5, 4.5),
                       'amplitude': random.uniform(0.25, 0.45)}

            cam  = attach_cam(hv)
            stk  = 0

            for fi in range(cfn):
                wd.tick()

                if bh == 'tailgating':
                    ok3 = tailgating_control(ego, tgc, tv2, dt)
                    if not ok3:
                        print(f"\n[Clip {ci:03d}] Actor destroyed at frame {fi} — ending clip.")
                        break
                    if fi > 120:
                        d2t = ego.get_location().distance(tv2.get_location())
                        if d2t > 100.0:
                            print(f"\n[Clip {ci:03d}] Target lost — ending clip.")
                            break
                        if speed(ego) < 0.3 and speed(tv2) > 2.0:
                            stk += 1
                        else:
                            stk = 0
                        if stk > 300:
                            print(f"\n[Clip {ci:03d}] Ego stuck for 5s — ending clip.")
                            break

                elif bh == 'swerving':
                    ss = swerve_control(ego, ss, dt=dt)

                if fi % 60 == 0:
                    sk = speed(hv) * 3.6
                    ex = ''
                    if bh == 'tailgating' and tv2 and tv2.is_alive:
                        rd2 = ego.get_location().distance(tv2.get_location())
                        gp2 = max(0.0, rd2 - 2 * VHL)
                        tk2 = speed(tv2) * 3.6
                        ex  = f'  gap:{gp2:4.1f}m  tgt:{tk2:4.0f}km/h'
                    print(f"\r  clip_{ci:03d}  {fi*dt:4.0f}s/{cs:.0f}s  "
                          f"frames:{cf[0]:5d}  spd:{sk:4.0f}km/h  "
                          f"q:{sq.qsize():3d}{ex}    ",
                          end='', flush=True)

            ftc  = cf[0]
            ts  += ftc
            print(f"\n[Clip {ci:03d}] Done — {ftc} raw frames saved.")

            cam.stop()
            cam.destroy()
            cam = None

            if bh in ('tailgating', 'swerving') and ego is not None:
                destroy_actor(ego)
                ego = None

            sq.join()

            if not args.no_thin:
                kp = thin_dir(cld, args.target_fps, args.src_fps)
                print(f"[Thin]  clip_{ci:03d}: {kp} frames kept "
                      f"({kp/args.target_fps:.1f}s @ {args.target_fps}fps)")

        print(f"\n[Recorder] All clips done. {ts} raw frames total.")

    except KeyboardInterrupt:
        print(f"\n[Recorder] Interrupted. {ts} raw frames saved so far.")

    finally:
        if cam is not None:
            try:
                cam.stop(); cam.destroy()
            except Exception:
                pass
        if ego is not None:
            destroy_actor(ego)
        se.set()
        flush_queue(sq, se, wf)
        wrt.join(timeout=10)
        print("[Cleanup] Destroying NPC actors...")
        try:
            for ct in nct:
                ct.stop()
        except Exception:
            pass
        ids = ([v.id for v in nvs if v and v.is_alive] +
               [w.id for w in nwl if w and w.is_alive] +
               [c.id for c in nct if c and c.is_alive])
        if ids:
            cl.apply_batch([carla.command.DestroyActor(i) for i in ids])
        wd.apply_settings(os2)
        tm.set_synchronous_mode(False)
        print("[Cleanup] Done.")

        print(f"\n{'='*62}")
        print(f"Dataset summary — {bh.upper()}")
        print(f"{'='*62}")
        ac2 = sorted(glob.glob(os.path.join(bd, 'clip_*')))
        tf2 = 0
        for cdd in ac2:
            fr2 = glob.glob(os.path.join(cdd, 'frame_*.png'))
            fl2 = args.target_fps if not args.no_thin else args.src_fps
            sc2 = len(fr2) / fl2
            print(f"  {os.path.basename(cdd):<12}  {len(fr2):5d} frames  ({sc2:.1f}s @ {fl2}fps)")
            tf2 += len(fr2)
        fl2 = args.target_fps if not args.no_thin else args.src_fps
        print(f"  {'TOTAL':<12}  {tf2:5d} frames  ({tf2/fl2:.1f}s @ {fl2}fps)")
        print(f"{'='*62}")


def main():
    p = argparse.ArgumentParser(
        description='CARLA Behavior Dataset Recorder — normal / tailgating / swerving')
    p.add_argument('--behavior',      choices=['normal', 'tailgating', 'swerving'])
    p.add_argument('--clips',         type=int,   default=10)
    p.add_argument('--clip-duration', type=float, default=60.0)
    p.add_argument('--target-fps',    type=int,   default=20)
    p.add_argument('--src-fps',       type=int,   default=60)
    p.add_argument('--host',          default='127.0.0.1')
    p.add_argument('--port',          type=int,   default=2000)
    p.add_argument('--res',           default='1280x720')
    p.add_argument('--num-vehicles',  type=int,   default=80)
    p.add_argument('--num-walkers',   type=int,   default=20)
    p.add_argument('--no-render',     action='store_true')
    p.add_argument('--no-thin',       action='store_true')
    args = p.parse_args()

    if not args.behavior:
        p.error("--behavior required  (normal / tailgating / swerving)")

    args.width, args.height = [int(x) for x in args.res.split('x')]
    record(args)


if __name__ == '__main__':
    main()
