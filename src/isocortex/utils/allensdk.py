import functools
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from allensdk.core.reference_space import ReferenceSpace
import nrrd


def get_reference_space(allen_dir):

    rspc = ReferenceSpaceCache(
        resolution=25,
        reference_space_key="annotation/ccf_2017",
        manifest=str(allen_dir / "manifest.json"),
    )
    return rspc.get_reference_space(
        structure_file_name=str(allen_dir / "structures.json"),
        annotation_file_name=str(allen_dir / "ccf_2017" / f"annotation_{25}.nrrd"),
    )


def save_masks(rsp, ids_to_save, mask_dir):

    mask_dir.mkdir(parents=True, exist_ok=True)
    writer = functools.partial(ReferenceSpace.check_and_write, str(mask_dir))
    for structure_id in rsp.many_structure_masks(ids_to_save, writer):
        print(f"  made mask for structure {structure_id}")


def create_reference_space_from_regions(rsp, acronyms, ids, mask_dir):

    print("Creating reference space from regions:", acronyms)

    masks = {}
    for sid in ids:
        f = mask_dir / f"structure_{sid}.nrrd"
        if not f.exists():
            raise FileNotFoundError(f"no mask for structure {sid} at {f}")
        masks[sid] = nrrd.read(str(f))[0]

    new_annotation = np.zeros(rsp.annotation.shape, dtype=np.int32)
    for sid, mask in masks.items():
        overlap = np.count_nonzero((mask > 0) & (new_annotation != 0))
        if overlap:
            print(f"  warning: {sid} overlaps {overlap} voxels already assigned")
        new_annotation[mask > 0] = sid

    present = set(np.unique(new_annotation)) - {0}
    missing = [sid for sid in ids if sid not in present]
    if missing:
        empty = [sid for sid in missing if not masks[sid].any()]
        raise ValueError(
            f"{len(missing)} region(s) absent from the annotation: {missing}\n"
            f"  empty masks: {empty}\n"
            f"  overwritten: {[s for s in missing if s not in empty]}"
        )

    assert len(acronyms) == len(present)
    return ReferenceSpace(
        structure_tree=rsp.structure_tree,
        annotation=new_annotation,
        resolution=rsp.resolution,
    )
