"""Schnakenberg reaction–diffusion and independent numerical controls."""

from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
J = np.array([[0.8, 1], [-1.8, -1.0]])


def initial(n, seed=0, smooth=False):
    rng = np.random.default_rng(seed)
    if smooth:
        x, y = np.meshgrid(
            np.arange(n) * 2 * np.pi / n, np.arange(n) * 2 * np.pi / n, indexing="ij"
        )
        fields = []
        for _ in range(2):
            f = sum(
                rng.normal() * np.cos(m * x + k * y + rng.uniform(0, 2 * np.pi))
                for m, k in [(1, 1), (2, 1), (3, 2), (4, 0), (0, 4), (3, 3)]
            )
            fields.append(f / f.std() * 0.001)
    else:
        fields = [rng.normal(size=(n, n)) for _ in range(2)]
        fields = [(f - f.mean()) / f.std() * 0.001 for f in fields]
    return 1 + fields[0], 0.9 + fields[1]


def qgrid(n):
    q = 2 * np.pi * np.fft.fftfreq(n, d=40 / n)
    return q[:, None] ** 2 + q[None, :] ** 2


def step(u, v, dt, du, dv, q2, reactions=True):
    f = 0.1 - u + u * u * v if reactions else 0
    h = 0.9 - u * u * v if reactions else 0
    return (
        np.fft.ifft2(np.fft.fft2(u + dt * f) / (1 + dt * du * q2)).real,
        np.fft.ifft2(np.fft.fft2(v + dt * h) / (1 + dt * dv * q2)).real,
    )


def wavelength(u):
    if u.std() < 1e-8:
        return None
    power = abs(np.fft.fft2(u - u.mean())) ** 2
    power[0, 0] = 0
    i = np.unravel_index(power.argmax(), power.shape)
    return float(2 * np.pi / np.sqrt(qgrid(len(u))[i]))


def simulate(dv, seed=0, n=64, dt=0.02, T=200, smooth=False, extend=True):
    u, v = initial(n, seed, smooth)
    q2 = qgrid(n)
    stride = round(1 / dt)
    rows = []
    snaps = {0: u.copy()}
    recent = {}
    horizon = T
    j = 0
    reason = ""
    settled = False
    while True:
        if j % stride == 0:
            time = round(j * dt, 8)
            assert (
                np.isfinite(u).all()
                and np.isfinite(v).all()
                and min(u.min(), v.min()) >= -1e-10
            )
            rows.append(
                {
                    "time": time,
                    "std_u": float(u.std()),
                    "mean_u": float(u.mean()),
                    "min_u": float(u.min()),
                    "min_v": float(v.min()),
                }
            )
            recent[time] = u.copy()
            if time - 26 in recent:
                del recent[time - 26]
            if time in [0, 25, 50, 100, 200, 400]:
                snaps[time] = u.copy()
            if time >= horizon:
                sd = u.std()
                window = [r["std_u"] for r in rows if r["time"] >= time - 25]
                change = float(np.sqrt(np.mean((u - recent[time - 25]) ** 2)))
                settled = bool(
                    sd < 1e-8 or (np.ptp(window) < 0.02 * sd and change < 0.02 * sd)
                )
                if horizon == 200 and extend and not settled:
                    horizon = 400
                    reason = "Final 25-unit window failed settling criteria"
                else:
                    break
        u, v = step(u, v, dt, 1, dv, q2)
        j += 1
    return (
        pd.DataFrame(rows),
        snaps,
        {
            "Dv": dv,
            "seed": seed,
            "grid": n,
            "dt": dt,
            "smooth_initial": smooth,
            "final_time": horizon,
            "final_std": float(u.std()),
            "wavelength": wavelength(u),
            "settled": settled,
            "extension_reason": reason,
        },
        u,
    )


def checks():
    n = 16
    q = qgrid(n)
    u = np.ones((n, n))
    v = 0.9 * u
    for _ in range(100):
        u, v = step(u, v, 0.02, 1, 10, q)
    eq = max(abs(u - 1).max(), abs(v - 0.9).max())
    assert eq < 1e-10
    u, v = initial(n)
    mean = u.mean()
    sd = u.std()
    for _ in range(100):
        u, v = step(u, v, 0.02, 1, 10, q, False)
    diffusion_error = float(abs(u.mean() - mean))
    assert diffusion_error < 1e-12 and u.std() < sd
    u = np.full((n, n), 1.01)
    v = np.full((n, n), 0.89)
    for _ in range(1000):
        u, v = step(u, v, 0.001, 0, 0, q)
    exact = solve_ivp(
        lambda t, y: [0.1 - y[0] + y[0] ** 2 * y[1], 0.9 - y[0] ** 2 * y[1]],
        [0, 1],
        [1.01, 0.89],
        rtol=1e-11,
        atol=1e-12,
    ).y[:, -1]
    err = float(max(abs(u[0, 0] - exact[0]), abs(v[0, 0] - exact[1])))
    assert err < 1e-4
    assert np.max(np.linalg.eigvals(J).real) < 0
    return {
        "uniform_equilibrium_error": float(eq),
        "reaction_only_error": err,
        "diffusion_mean_error": diffusion_error,
    }


def main():
    OUT.mkdir(exist_ok=True)
    validation = checks()
    rows = []
    series = []
    saved = {}
    for dv in [1, 5, 10]:
        for seed in range(5):
            ts, snap, r, u = simulate(dv, seed)
            rows.append(r)
            ts["Dv"] = dv
            ts["seed"] = seed
            series.append(ts)
            if seed == 0:
                saved[dv] = (snap, r, u)
            print(dv, seed, r["final_std"], r["settled"], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "run-summary.csv", index=False)
    pd.concat(series).to_csv(OUT / "spatial-variation.csv", index=False)
    base = saved[10][1]
    _, _, fine, _ = simulate(10, dt=0.01, T=base["final_time"], extend=False)
    ref = []
    _, _, a, _ = simulate(10, smooth=True, T=400, extend=False)
    _, _, b, _ = simulate(10, n=128, smooth=True, T=400, extend=False)
    for name, x, y in [("time step", base, fine), ("spatial grid", a, b)]:
        sd = abs(x["final_std"] - y["final_std"]) / max(y["final_std"], 1e-12)
        wave = (
            abs(x["wavelength"] - y["wavelength"]) / y["wavelength"]
            if x["wavelength"] and y["wavelength"]
            else None
        )
        ref.append(
            {
                "check": name,
                "relative_std_difference": sd,
                "relative_wavelength_difference": wave,
                "passed": bool(sd < 0.1 and (wave is None or wave < 0.2)),
                "coarse": x,
                "fine": y,
            }
        )
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    k = np.linspace(0, 1.2, 301)
    disp = []
    for dv in [1, 5, 10]:
        growth = np.array(
            [np.linalg.eigvals(J - z * z * np.diag([1, dv])).real.max() for z in k]
        )
        axs[0].plot(k, growth, label=f"Dv={dv}")
        disp.extend(
            {"Dv": dv, "k": float(z), "growth": float(g)} for z, g in zip(k, growth)
        )
        for ts in series:
            if ts.Dv.iloc[0] == dv:
                axs[1].semilogy(
                    ts.time,
                    np.maximum(ts.std_u, 1e-16),
                    alpha=0.65,
                    color={1: "tab:blue", 5: "tab:orange", 10: "tab:green"}[dv],
                    label=f"Dv={dv}" if ts.seed.iloc[0] == 0 else None,
                )
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_ylim(-0.6, 0.12)
    axs[0].set_xlabel("Angular wavenumber k")
    axs[0].set_ylabel("Largest linear growth rate")
    axs[0].legend()
    discrete = np.unique(qgrid(64))
    unstable = discrete[(discrete > 0.2) & (discrete < 0.5)]
    axs[0].scatter(
        np.sqrt(unstable),
        np.zeros(len(unstable)),
        s=10,
        c="k",
        marker="|",
        label="Supported unstable modes",
    )
    axs[1].set_xlabel("Model time")
    axs[1].set_ylabel("Spatial SD of u")
    axs[1].legend()
    im = axs[2].imshow(saved[10][2], origin="lower", extent=(0, 40, 0, 40))
    axs[2].set_title("Dv10, seed0, t=" + str(base["final_time"]))
    axs[2].set_xlabel("Model space")
    fig.colorbar(im, ax=axs[2], label="u")
    fig.tight_layout()
    fig.savefig(OUT / "stability-and-pattern.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(disp).to_csv(OUT / "dispersion.csv", index=False)
    fig, axs = plt.subplots(3, 5, figsize=(13, 8))
    for i, dv in enumerate([1, 5, 10]):
        for j, t in enumerate([0, 25, 50, 100, 200]):
            field = saved[dv][0][t]
            im = axs[i, j].imshow(
                field, origin="lower", vmin=0.5, vmax=1.5, cmap="viridis"
            )
            axs[i, j].set_title(f"Dv{dv}, t={t}")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    fig.suptitle("Seed0 fields, common concentration scale 0.5–1.5")
    fig.tight_layout()
    fig.savefig(OUT / "pattern-development.png", dpi=160)
    plt.close(fig)
    record = {
        "checks": validation,
        "refinement": ref,
        "supported_unstable_k_squared": unstable.tolist(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
