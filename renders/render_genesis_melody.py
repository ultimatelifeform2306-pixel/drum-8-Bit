# Re-render the clip's melody with Genesis-style FM (YM2612-like 4-op voices).
import numpy as np, wave
SR = 44100
DUR = 11.363
T0 = 2.835          # first G4 in the video timeline
CYC = 3.35          # 2 bars at ~143 BPM
SD = CYC / 32       # one 16th note
# (start cell, length in cells, midi)
PHRASE = [(0,6,67),(6,4,75),(10,1,72),(11,1,71),(12,1,69),(13,2,66),(15,1,64),
          (16,5,58),(21,1,57),(22,7,60),(29,3,66)]
out = np.zeros(int(SR*DUR)+SR)
mf = lambda m: 440*2**((m-69)/12) * 2**(0.2/12)   # clip is tuned ~20 cents sharp

def env(n, a, d, s, r, gate):
    t = np.arange(n)/SR
    e = np.where(t < a, t/a, s + (1-s)*np.exp(-(t-a)/max(d,1e-4)))
    rel = t > gate
    e[rel] = e[rel] * np.exp(-(t[rel]-gate)/r)
    return e

def fm_lead(f, dur, vel=1.0):
    """Genesis 'brass lead' patch: op1 (with feedback) -> op2 carrier, plus a detuned op3->op4 pair."""
    n = int(SR*(dur+0.25)); t = np.arange(n)/SR
    vib = 1 + 0.010*np.sin(2*np.pi*5.6*t)*np.clip((t-0.15)/0.2, 0, 1)
    ph = 2*np.pi*np.cumsum(f*vib)/SR
    # op1 with self-feedback (sawtooth-ish), as on the YM2612
    m = np.zeros(n); prev = 0.0; fb = 0.9
    idx1 = 1.6 + 1.6*np.exp(-t/0.08)
    s1 = ph
    # vectorised feedback approximation: two-pass
    m = np.sin(s1 + fb*np.sin(s1))
    car = np.sin(ph + idx1*m)
    ph2 = 2*np.pi*np.cumsum(f*1.004*vib)/SR
    car2 = np.sin(ph2 + (1.2+1.0*np.exp(-t/0.1))*np.sin(2*ph2))
    e = env(n, 0.006, 0.25, 0.75, 0.06, dur)
    return vel*e*(0.62*car + 0.38*car2)

def fm_bass(f, dur):
    n = int(SR*(dur+0.2)); t = np.arange(n)/SR
    ph = 2*np.pi*f*t
    car = np.sin(ph + (2.2*np.exp(-t/0.12)+0.7)*np.sin(ph))
    return env(n, 0.003, 0.4, 0.6, 0.05, dur)*car

def place(sig, start, gain):
    i = int(start*SR); out[i:i+len(sig)] += gain*sig[:len(out)-i]

# melody, two passes like the video
for c in range(2):
    base = T0 + c*CYC
    for st, ln, m in PHRASE:
        place(fm_lead(mf(m), ln*SD - 0.012), base + st*SD, 0.42)
# pickup F#4 into the first G (heard at ~2.74s), and the next G at the cut
place(fm_lead(mf(66), 0.08), T0 - SD, 0.3)
end = T0 + 2*CYC
place(fm_lead(mf(67), 0.12), end, 0.3)

# D drone underneath, pulsed in 8ths on an FM bass like the original's pedal
for k in range(int(2*CYC/(2*SD))):
    tt = T0 + k*2*SD
    place(fm_bass(mf(38), 2*SD*0.8), tt, 0.30 if k % 2 == 0 else 0.20)
# soft sustained D3+A3 pad (FM, slow attack)
n = int(SR*2*CYC); t = np.arange(n)/SR
pad = sum(np.sin(2*np.pi*mf(m)*t + 0.5*np.sin(2*np.pi*mf(m)*t)) for m in (50, 57))
pe = np.minimum(1, t/0.4) * np.minimum(1, (2*CYC - t)/0.3)
place(pad*pe, T0, 0.07)

# echo like Genesis games did with a spare channel (dotted-8th, -10 dB)
dly = int(SR*SD*3)
echo = np.zeros_like(out); echo[dly:] = out[:-dly]*0.14
out = out + echo
# YM2612-ish output: 9-bit DAC and a gentle low-pass (model 1 filter)
out = out[:int(SR*DUR)]
out = np.round(out*256)/256
a = np.exp(-2*np.pi*7000/SR); y = np.zeros_like(out); p = 0.0
for i in range(len(out)):
    p = (1-a)*out[i] + a*p; y[i] = p
y /= max(1e-9, np.abs(y).max()) / 0.89
st = np.stack([y, y], 1)
w = wave.open('v2/genesis_melody.wav', 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((st*32767).astype('<i2').tobytes()); w.close()
print('ok', len(y)/SR)
