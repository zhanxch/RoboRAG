"""Trajectory-only grasp verification; this does not claim tactile contact sensing."""
import numpy as np

def grasp_metrics(states,actions,start_frame=0,horizon=80,hold_steps=25,terminated=False):
    s=np.asarray(states);a=np.asarray(actions)
    point=s[1:,29:32]*.1;tcp=s[1:,14:17]*.1
    lift=point[:,2];gap=np.linalg.norm(point-tcp,axis=1)
    actual_open=(s[1:,6]+1)/2;command_open=(a[:,6]+1)/2
    held=(lift>=.005)&(gap<.055)&(actual_open<.35)&(command_open<.35)
    hold=0;max_hold=0
    for ok in held:hold=hold+1 if ok else 0;max_hold=max(hold,max_hold)
    tail=s[-hold_steps:];rel=(tail[:,29:32]-tail[:,14:17])*.1
    slip=float(np.linalg.norm(rel-rel[0],axis=1).max())
    low=int(np.argmin(s[:,16]));tcp_rise=float((s[-1,16]-s[low,16])*.1)
    cloth_rise=float((s[-1,31]-s[low,31])*.1)
    ratio=cloth_rise/max(tcp_rise,.005)
    stable=bool(hold>=hold_steps and slip<=.015 and ratio>=.5 and len(a)+start_frame==horizon and not terminated)
    return dict(success=stable and start_frame==0,stable_retention=stable,
        final_hold_steps=int(hold),max_hold_steps=int(max_hold),relative_slip_m=slip,
        lift_follow_ratio=ratio,max_lift_m=float(lift.max()),final_gap_m=float(gap[-1]),
        verification='trajectory proxy: elevation, co-motion, physical finger closure and relative slip; not contact-force sensing')
