#!/usr/bin/env python3
"""Check that dataset folders and files match what Train.py and Evaluation.py expect."""

import os

# What Train.py / Evaluation.py expect (from data_set/)
EXPECTED = {
    "LFW": {
        "root": "LFW/lfw_align_112",
        "file_list": "LFW/pairs.txt",
    },
    "CFP-FP": {
        "root": "CFP-FP/CFP_FP_aligned_112",
        "file_list": "CFP-FP/cfp_fp_pair.txt",
    },
    "AgeDB-30": {
        "root": "AgeDB-30/agedb30_align_112",
        "file_list": "AgeDB-30/agedb_30_pair.txt",
    },
    "faces_emore (train)": {
        "root": "faces_emore_images",
        "file_list": "faces_emore_images/faces_emore_align_112.txt",
    },
    "CASIA (train)": {
        "root": "CASIA_Webface_Image",
        "file_list": "CASIA_Webface_Image/webface_align_112.txt",
    },
}

# Alternative names the code also accepts (Train.py checks these)
CASIA_ROOT_ALTS = ("CASIA", "CASIA_Webface_Image")
CASIA_LIST_NAME = "webface_align_112.txt"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    all_ok = True
    for name, paths in EXPECTED.items():
        root_path = paths["root"]
        list_path = paths["file_list"]
        # CASIA: accept either CASIA or CASIA_Webface_Image
        if "CASIA" in name:
            root_ok = any(os.path.isdir(d) for d in CASIA_ROOT_ALTS)
            if root_ok:
                actual_root = next(d for d in CASIA_ROOT_ALTS if os.path.isdir(d))
                list_path = os.path.join(actual_root, CASIA_LIST_NAME)
            list_ok = os.path.isfile(list_path)
        else:
            root_ok = os.path.isdir(root_path)
            list_ok = os.path.isfile(list_path)
        status = "OK" if (root_ok and list_ok) else "MISSING"
        if not (root_ok and list_ok):
            all_ok = False
        root_status = "dir exists" if root_ok else "dir missing"
        list_status = "file exists" if list_ok else "file missing"
        print(f"  {name}: {status}")
        if "CASIA" in name and root_ok:
            print(f"    -> {actual_root}/ ({root_status})")
        else:
            print(f"    -> {root_path} ({root_status})")
        print(f"    -> {list_path} ({list_status})")
    eval_names = ["LFW", "CFP-FP", "AgeDB-30"]
    eval_ok = all(
        os.path.isdir(EXPECTED[n]["root"]) and os.path.isfile(EXPECTED[n]["file_list"])
        for n in eval_names
    )
    print()
    if eval_ok:
        print("Eval datasets (LFW, CFP-FP, AgeDB-30) are in the right spot for Train.py / Evaluation.py.")
    casia_ok = any(
        os.path.isdir(d) and os.path.isfile(os.path.join(d, CASIA_LIST_NAME))
        for d in CASIA_ROOT_ALTS
    )
    faces_ok = os.path.isdir(EXPECTED["faces_emore (train)"]["root"]) and os.path.isfile(EXPECTED["faces_emore (train)"]["file_list"])
    if not casia_ok and not faces_ok:
        print("Training data (faces_emore or CASIA) is missing; needed for Train.py.")
    # Hint if CASIA dir exists but annotation file missing
    for d in CASIA_ROOT_ALTS:
        if os.path.isdir(d) and not os.path.isfile(os.path.join(d, CASIA_LIST_NAME)):
            print("To generate CASIA annotation file, run from data_set/: python anno_generation.py --root", d, "--file", os.path.join(d, CASIA_LIST_NAME))
            break
    return 0 if eval_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
