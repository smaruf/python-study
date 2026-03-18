# 01 — Structural Implementation

Structural implementation is the foundation of every aircraft build.
A poorly designed structure leads to in-flight failure regardless of how good the avionics are.

---

## 1. Airframe Taxonomy

### 1.1 Fixed-Wing Structures

```
Fixed-Wing Airframe
│
├── Wing assembly
│   ├── Main spar (primary load path)
│   ├── Rear spar / spar box
│   ├── Wing ribs (aerofoil shape holders)
│   ├── Leading-edge (LE) sheeting
│   ├── Trailing-edge (TE) strip
│   ├── Control surfaces (aileron, flap, spoiler)
│   └── Winglets / tip tanks (optional)
│
├── Fuselage
│   ├── Nose section (avionics bay)
│   ├── Centre section (battery / payload bay)
│   ├── Tail boom / tail section
│   └── Hatch / access panels
│
├── Empennage (tail)
│   ├── Horizontal stabiliser + elevator
│   ├── Vertical fin + rudder
│   └── V-tail / T-tail (variants)
│
└── Undercarriage
    ├── Nose gear / tail-dragger gear
    └── Retractable gear (production)
```

### 1.2 Multi-Rotor Structures

```
Multi-Rotor Frame
│
├── Centre plate (top + bottom)
│   ├── FC mounting (vibration-isolated)
│   ├── ESC stack / power distribution board (PDB)
│   └── Payload / battery bay
│
├── Arms (4 / 6 / 8 for quad / hex / octo)
│   ├── Carbon-fibre tube or injection-moulded
│   ├── Motor mounts at tips
│   └── Fold mechanism (optional, for portability)
│
└── Landing gear
    ├── Fixed skids
    ├── Retractable legs (production)
    └── Dampened feet (camera / survey drones)
```

---

## 2. Materials

| Material | Density (g/cm³) | Tensile Strength (MPa) | Typical Use |
|---|---|---|---|
| Foam (EPO/EPP) | 0.03–0.06 | 0.2–0.5 | Trainer, budget RC bodies |
| Balsa wood | 0.12–0.20 | 25–35 | Traditional RC ribs/spars |
| PLA (3D-printed) | 1.24 | 50 | Non-structural covers, mounts |
| PETG (3D-printed) | 1.27 | 53 | Light structural parts |
| Nylon / PA12 | 1.15 | 55 | Flex hinges, gear, prop adapters |
| CF-Nylon (3D-printed) | 1.20 | 80 | Structural frames, arm tubes |
| Fibreglass (GFRP) | 1.80 | 350 | Fuselage skins, flat plates |
| Carbon Fibre (CFRP) | 1.60 | 600 | Spars, tubes, high-perf frames |
| Aluminium 6061 | 2.70 | 276 | Motor mounts, hubs, machined parts |
| Titanium Ti-6Al-4V | 4.43 | 950 | Fasteners, combat-grade frames |

**Selection heuristic:**
- Hobby / simple → foam body + 3D-printed mounts
- Medium / prosumer → CFRP frame + CF-Nylon arms
- Production → full CFRP / aluminium monocoque + titanium fasteners

---

## 3. Load Paths and Structural Analysis

### 3.1 Primary Load Cases

| Load case | Source | Critical component |
|---|---|---|
| Bending (span-wise) | Aerodynamic lift | Main spar |
| Torsion | Aileron deflection, gusts | Spar box / D-box |
| Axial compression | Motor thrust | Fuselage centre section |
| Vibration (HF) | Motor/prop imbalance | Frame arms, FC mount |
| Impact | Crash landing | Landing gear, nose |
| Thermal cycling | Electronics heat, sun | Composite delamination risk |

### 3.2 Key Structural Formulae

**Wing root bending moment**
```
M = (L/2) × (b/4)
```
where `L` = total lift force (≈ all-up weight × load factor), `b` = wingspan.

**Spar bending stress** (from `analysis/stress.py`):
```python
sigma = bending_stress(force, length, second_moment_of_area)
# sigma = (F × L) / I
```

**Safety factor** — target a minimum factor of **2.5** for hobby, **4.0** for production:
```
SF = Material_Yield_Strength / Calculated_Stress
```

### 3.3 Finite Element Analysis (FEA) Workflow

1. Model geometry in FreeCAD or Fusion 360
2. Assign material properties
3. Apply load cases (distributed lift, concentrated thrust, point impacts)
4. Run solver (CalculiX for open-source FEA)
5. Inspect stress concentrations → redesign radius/fillets
6. Iterate until SF ≥ target

---

## 4. Wing Structural Design

### 4.1 Spar Sizing

```
Required spar section modulus:
Z = M / sigma_allow

For a rectangular spar (width w, height h):
Z = (w × h²) / 6

For a hollow CFRP tube (outer d_o, inner d_i):
I = π/64 × (d_o⁴ − d_i⁴)
Z = I / (d_o/2)
```

### 4.2 Rib Spacing

- Foam ribs: 80–120 mm apart
- Balsa ribs: 50–80 mm apart
- Carbon composite ribs: 100–200 mm apart

### 4.3 Control Surface Hinge Lines

- Hinge at 25–30% of chord from trailing edge
- Use piano wire (1.5–2 mm) for hobby, machined aluminium brackets for production
- Minimum three hinge points per surface to distribute load

---

## 5. Multi-Rotor Frame Sizing

### 5.1 Motor-to-Motor Distance (Wheelbase)

```
Wheelbase (mm) = Prop_diameter_inches × 25.4 × 1.3
```
Example: 5-inch props → wheelbase ≈ 165 mm (standard 5" freestyle quad)

### 5.2 Arm Cross-Section Selection

| Payload class | All-up weight | Arm tube (CFRP) |
|---|---|---|
| Micro | < 250 g | 8 × 1 mm wall |
| Mini | 250–500 g | 10 × 1 mm wall |
| Standard | 500 g–2 kg | 12 × 1.5 mm wall |
| Heavy lift | 2–10 kg | 16 × 2 mm wall |
| Production cargo | > 10 kg | 20 × 2 mm wall + monocoque |

### 5.3 Vibration Isolation

- Use silicone grommets (Shore 30–40) between FC and frame
- Flight controller mounting holes: 30.5 × 30.5 mm (standard) or 20 × 20 mm (mini)
- Add dampening foam pads under ESC stack

---

## 6. Structural Integration Checklist

Before the first flight, verify:

- [ ] Spar passes bending stress calculation (SF ≥ 2.5)
- [ ] All CF tubes inspected for delamination under strong light
- [ ] Motor mounts torque-tested to 1.5× max motor thrust
- [ ] Control surface hinges flex freely with zero slop
- [ ] Battery/payload mass accounted for in CG calculation
- [ ] Landing gear rated for drop-test load (≥ 3× AUW over 30 cm drop)
- [ ] All fasteners thread-locked (Loctite 243 or equivalent)
- [ ] Frame resonant frequency measured — not within 20% of motor idle frequency

---

## 7. References & Further Reading

- Raymer, D. *Aircraft Design: A Conceptual Approach* — Chapter 8 (Structures)
- Sighard Hoerner, *Fluid Dynamic Drag* — aerodynamic load estimation
- ArduPilot frame design guide: <https://ardupilot.org/copter/docs/build-your-own-multicopter.html>
- PX4 Developer Guide (hardware): <https://docs.px4.io/main/en/hardware/>
- OpenVSP (free parametric aircraft modeller): <https://openvsp.org>
