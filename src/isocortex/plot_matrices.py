import re
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go

TABLE_PATH = Path("./data/tables/atlas.xlsx")
LOCAL_TABLE_PATH = Path("./data/tables/local.xlsx")

REGIONS = [
    "FRP",
    "MO",
    "SS",
    "GU",
    "VISC",
    "AUD",
    "VIS",
    "ACA",
    "PL",
    "ILA",
    "ORB",
    "AI",
    "RSP",
    "PTLp",
    "TEa",
    "PERI",
    "ECT",
]

def load_table(fpath, sheet_name):
    print(f"Loading: {fpath} [{sheet_name}]")

    if not fpath.exists():
        raise FileNotFoundError(f"File does not exist: {fpath}")
    try:
        return pd.read_excel(fpath, sheet_name=sheet_name, header=None)
    except ValueError as e:
        raise ValueError(f"Sheet '{sheet_name}' not found in {fpath}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}") from e


def pivot_and_detect(df, keep_keys):
    df_pivot = df.copy()
    df_pivot.columns = df_pivot.iloc[0]
    df_pivot = df_pivot.drop(df_pivot.index[0])
    df_pivot.index = df_pivot.iloc[:, 0]
    df_pivot = df_pivot.drop(df_pivot.columns[0], axis=1)

    df_pivot = df_pivot.apply(
        lambda col: (
            pd.to_numeric(col, errors="coerce").fillna(col)
            if pd.to_numeric(col, errors="coerce").notna().any()
            else col
        )
    )

    def matches(label):
        return any(k in str(label) for k in keep_keys)

    mask = df_pivot.index.to_series().apply(matches)
    df_pruned = df_pivot.loc[mask, mask]

    header_idx = None
    division_row = None
    for idx, row in df_pruned.iterrows():
        vals = row.dropna().astype(str)
        if (
            header_idx is None
            and sum(bool(re.fullmatch(r"[A-Z0-9\-]+", v)) for v in vals) > 40
        ):
            header_idx = idx
        if (
            division_row is None
            and sum(bool(re.fullmatch(r"[A-Z]{2,6}", v)) for v in vals) > 5
        ):
            division_row = idx

    hem_row = None
    if header_idx is not None:
        idx_list = list(df_pruned.index)
        i = idx_list.index(header_idx)
        if i + 1 < len(idx_list):
            hem_row = idx_list[i + 1]

    return df_pruned, (division_row, header_idx, hem_row)


def prepare_pair(connectivity_raw, distances_raw, regions):
    connectivity_df, _ = pivot_and_detect(connectivity_raw, regions)
    distances_df, _ = pivot_and_detect(distances_raw, regions)

    if list(distances_df.index) != list(connectivity_df.index) or list(
        distances_df.columns
    ) != list(connectivity_df.columns):
        print("  note: distance labels differ from connectivity labels; reindexing")
        distances_df = distances_df.reindex(
            index=connectivity_df.index, columns=connectivity_df.columns
        )

    return connectivity_df, distances_df


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_data_matrix(
    connectivity_df,
    distance_df,
    distance=False,
    title="",
    color_bar_title="",
    cmap="Plasma",
    cell_size=30,
):
    """Heatmap of one matrix, with the other shown in the hover text.

    distance=False colors by connectivity; distance=True colors by distance.
    """
    primary, secondary = (
        (distance_df, connectivity_df) if distance else (connectivity_df, distance_df)
    )

    sources = connectivity_df.index.tolist()
    targets = connectivity_df.columns.tolist()

    total_in = primary.sum(axis=0)
    total_out = primary.sum(axis=1)

    hover_text = []
    for src in sources:
        row = []
        for tgt in targets:
            dist = (primary if distance else secondary).loc[src, tgt]
            weight = (secondary if distance else primary).loc[src, tgt]

            if distance:
                text = (
                    f"Source: {src}<br>"
                    f"Target: {tgt}<br>"
                    f"Distance: {dist:.3f}<br>"
                    f"Weight: {weight:.3f}"
                )
            else:
                text = (
                    f"Source: {src}<br>"
                    f"Target: {tgt}<br>"
                    f"Weight: {weight:.3f}<br>"
                    f"Total out (source): {total_out[src]:.3f}<br>"
                    f"Total in (target): {total_in[tgt]:.3f}<br>"
                    f"Distance: {dist:.3f}"
                )
            row.append(text)
        hover_text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=primary.values,
            x=targets,
            y=sources,
            colorscale=cmap,
            colorbar=dict(title=color_bar_title),
            text=hover_text,
            hoverinfo="text",
        ),
    )

    margin = dict(l=120, r=150, t=150, b=60)
    font_size = max(16, int(cell_size * 0.5))

    fig.update_layout(
        xaxis=dict(side="top", tickfont=dict(size=font_size)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=font_size)),
        width=len(targets) * cell_size + margin["l"] + margin["r"],
        height=len(sources) * cell_size + margin["t"] + margin["b"],
        margin=margin,
        title=dict(text=title, font=dict(size=font_size * 2), pad=dict(b=40)),
        font=dict(size=font_size),
    )
    fig.update_traces(
        colorbar=dict(
            title=dict(text=color_bar_title, font=dict(size=font_size)),
            tickfont=dict(size=font_size),
        )
    )
    return fig


def emit(fig, out_path, save=True):
    """Save the figure if a path is given, otherwise show it."""
    if out_path and save:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".html":
            fig.write_html(str(out_path))
        else:
            fig.write_image(str(out_path))  # needs kaleido
        print(f"wrote {out_path}")
    else:
        fig.show()


# --------------------------------------------------------------------------
def main(
    # --- inputs ---
    table_path=TABLE_PATH,
    local_table_path=LOCAL_TABLE_PATH,
    regions=REGIONS,
    connectivity_sheet="connectivity",
    distances_sheet="distances",
    save=True,  # False -> show figures, write nothing
    connectivity_html=None,  # if set, save instead of showing
    distance_html=None,  # if set, save instead of showing
    # --- appearance ---
    cmap="Plasma",
    cell_size=30,
    connectivity_title="Empirical Connectivity Matrix",
    distance_title=None,  # default depends on use_local
    color_bar_title="Connection Strength",
):
    distance_title = "Distance Matrix"

    # --- atlas matrices ----------------------------------------------------
    connectivity_df, distances_df = prepare_pair(
        load_table(table_path, connectivity_sheet),
        load_table(table_path, distances_sheet),
        regions,
    )
    print(f"connectivity matrix: {connectivity_df.shape}")

    connectivity_fig = plot_data_matrix(
        connectivity_df=connectivity_df,
        distance_df=distances_df,
        distance=False,
        title=connectivity_title,
        color_bar_title=color_bar_title,
        cmap=cmap,
        cell_size=cell_size,
    )
    emit(connectivity_fig, connectivity_html, save)

    second_connectivity_df, second_distances_df = connectivity_df, distances_df

    distance_fig = plot_data_matrix(
        connectivity_df=second_connectivity_df,
        distance_df=second_distances_df,
        distance=True,
        title=distance_title,
        color_bar_title=color_bar_title,
        cmap=cmap,
        cell_size=cell_size,
    )
    emit(distance_fig, distance_html, save)

    return connectivity_fig, distance_fig


if __name__ == "__main__":
    main()
