import plotly.graph_objs as go
import numpy as np
from skimage import measure
import plotly.express as px


def plot_surfaces_and_injections(
    surfaces,
    id_to_color,
    id_to_acronym=None,
    width=800,
    height=800,
    id_to_opacity=None,
    default_opacity=0.9,
    camera=None,
    camera_eye=None,
    camera_center=None,
    camera_up=None,
    injection_coordinate_experiment_id_tuples=None,
    annotation_marker_size=5,
    annotation_marker_opacity=0.9,
    # ---- NEW ----
    top_outline=True,
    top_outline_width=3,
):
    surfaces = np.asarray(surfaces)
    n_surfaces, cor_dim, sag_dim, lr_dim = surfaces.shape

    # ----------------------------
    # 1) Build label volume for fast "hit region" detection
    # ----------------------------
    hit_region_ids = set()

    if injection_coordinate_experiment_id_tuples:
        label_vol = np.zeros((cor_dim, sag_dim, lr_dim), dtype=np.int32)
        for s in range(n_surfaces):
            vol = surfaces[s]
            m = vol != 0
            if m.any():
                label_vol[m] = int(vol[m][0])

        pts_plot = np.asarray(
            [c for c, _ in injection_coordinate_experiment_id_tuples], dtype=np.float32
        )

        x_lr = np.rint(pts_plot[:, 1]).astype(np.int32)
        y_sag = np.rint(-pts_plot[:, 2]).astype(np.int32)
        z_cor = np.rint(-pts_plot[:, 0]).astype(np.int32)

        in_bounds = (
            (z_cor >= 0)
            & (z_cor < cor_dim)
            & (y_sag >= 0)
            & (y_sag < sag_dim)
            & (x_lr >= 0)
            & (x_lr < lr_dim)
        )

        if np.any(in_bounds):
            hit_ids = label_vol[z_cor[in_bounds], y_sag[in_bounds], x_lr[in_bounds]]
            hit_ids = np.unique(hit_ids)
            hit_ids = hit_ids[hit_ids != 0]
            hit_region_ids = set(map(int, hit_ids))

        del label_vol

    # ----------------------------
    # Helper: silhouette edges from a fixed top view direction
    # ----------------------------
    def _top_silhouette_edges(x, y, z, faces, viewdir=(0.0, 0.0, -1.0)):


        f = faces.astype(np.int64, copy=False)
        v0 = np.stack([x[f[:, 0]], y[f[:, 0]], z[f[:, 0]]], axis=1)
        v1 = np.stack([x[f[:, 1]], y[f[:, 1]], z[f[:, 1]]], axis=1)
        v2 = np.stack([x[f[:, 2]], y[f[:, 2]], z[f[:, 2]]], axis=1)

        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n, axis=1)
        n_norm[n_norm == 0] = 1.0
        n = n / n_norm[:, None]

        vd = np.asarray(viewdir, dtype=np.float32)
        d = n @ vd  # face "facingness" vs viewdir

        # Build edge -> adjacent faces map
        edge_to_faces = {}
        for fi, (a, b, c) in enumerate(f):
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                edge_to_faces.setdefault((u, v), []).append(fi)

        silhouette = []
        for (u, v), adj in edge_to_faces.items():
            if len(adj) == 1:
                # boundary edge (open surface) -> draw it
                silhouette.append((u, v))
            else:
                f1, f2 = adj[0], adj[1]
                # silhouette if one face is front-facing and the other back-facing
                if (d[f1] >= 0) != (d[f2] >= 0):
                    silhouette.append((u, v))
        return silhouette

    # ----------------------------
    # 2) Build mesh traces
    # ----------------------------
    traces = []

    for s in range(n_surfaces):
        vol = surfaces[s]
        m = vol != 0
        if not m.any():
            continue

        region_id = int(vol[m][0])

        mask_u8 = m.astype(np.uint8, copy=False)
        verts, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)

        # verts are (z, y, x) = (cor, sag, LR)
        z_idx, y_idx, x_idx = verts.T

        x0 = -z_idx
        y0 = x_idx
        z0 = -y_idx

        i, j, k = faces.T

        rgb = id_to_color.get(region_id, (200, 200, 200))
        color_str = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        if region_id in hit_region_ids and region_id != 997:
            opacity = 1.0
        else:
            opacity = (
                id_to_opacity.get(region_id, default_opacity)
                if id_to_opacity
                else default_opacity
            )

        if id_to_acronym is not None:
            acronym = id_to_acronym.get(region_id, str(region_id))
            hovertemplate = f"Region: {acronym}<extra></extra>"
        else:
            hovertemplate = "Region: %{customdata}<extra></extra>"

        traces.append(
            go.Mesh3d(
                x=x0,
                y=y0,
                z=z0,
                i=i,
                j=j,
                k=k,
                color=color_str,
                opacity=opacity,
                showscale=False,
                customdata=np.array([region_id], dtype=np.int32),
                hovertemplate=hovertemplate,
            )
        )

        if top_outline:
            sil_edges = _top_silhouette_edges(
                x0, y0, z0, faces, viewdir=(0.0, 0.0, -1.0)
            )

            if sil_edges:
                xs, ys, zs = [], [], []
                eps = 1e-3  # tiny lift to reduce z-fighting
                for a, b in sil_edges:
                    xs.extend([x0[a], x0[b], None])
                    ys.extend([y0[a], y0[b], None])
                    zs.extend([z0[a] + eps, z0[b] + eps, None])

                traces.append(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color="black", width=top_outline_width),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # ----------------------------
    # 3) Injection points (already in plot coords)
    # ----------------------------
    if injection_coordinate_experiment_id_tuples:
        pts = np.asarray(
            [c for c, _ in injection_coordinate_experiment_id_tuples], dtype=np.float32
        )
        labels = [str(l) for _, l in injection_coordinate_experiment_id_tuples]

        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=annotation_marker_size,
                    opacity=annotation_marker_opacity,
                    color="rgb(255,0,0)",
                ),
                text=labels,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)

    # ----------------------------
    # 4) Camera + layout
    # ----------------------------
    cam = dict(camera) if camera else {}
    if camera_eye:
        cam["eye"] = dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])
    if camera_center:
        cam["center"] = dict(x=camera_center[0], y=camera_center[1], z=camera_center[2])
    if camera_up:
        cam["up"] = dict(x=camera_up[0], y=camera_up[1], z=camera_up[2])

    fig.update_layout(
        width=width,
        height=height,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=cam if cam else None,
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def plot_flat_hover_map(
    flat,
    structure_tree,
    show_boundaries=False,
    width=900,
    height=900,
    cmap=None,
):
    # --- Color mapping ---
    ids = np.unique(flat)
    ids = ids[ids != 0]  # exclude background

    H, W = flat.shape

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for sid, color in cmap.items():
        mask = flat == sid
        if sid == 0:
            rgba[mask] = (0, 0, 0, 0)
        else:
            rgba[mask] = (*color, 255)
    fig = px.imshow(rgba, binary_string=True)

    # --- Hover labels ---
    tree = structure_tree
    id_to_label = {s["id"]: f"{s['acronym']} ({s['name']})" for s in tree.nodes()}
    # id_to_label[0] = "Background"

    hover_labels = np.full(flat.shape, "", dtype=object)

    unique_ids = np.unique(flat)
    for sid in unique_ids:
        if sid == 0:
            continue
        hover_labels[flat == sid] = id_to_label.get(int(sid), f"ID {int(sid)}")

    fig.update_traces(
        customdata=hover_labels,
        hovertemplate="%{customdata}<extra></extra>",
    )

    # --- Optional boundaries ---
    if show_boundaries:
        unique_ids = [sid for sid in np.unique(flat) if sid != 0]

        for sid in unique_ids:
            region = (flat == sid).astype(float)
            if region.sum() == 0:
                continue

            contours = measure.find_contours(region, 0.5)
            for contour in contours:
                y, x = contour[:, 0], contour[:, 1]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color="black", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        coloraxis_showscale=False,
        showlegend=False,
        width=width,
        height=height,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.2,
        gridcolor="rgba(255,255,255,0.15)",  # light grid
        tick0=0,
        dtick=5,
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.2,
        gridcolor="rgba(255,255,255,0.15)",
        tick0=0,
        dtick=5,
        autorange="reversed",
    )

    fig.show()
