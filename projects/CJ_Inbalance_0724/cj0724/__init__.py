"""I/Q interface comparison base for the 0716 compound-jamming data."""

CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
# Active data live under E:\LJQ_code.  The former C: path is historical and
# unavailable on this workstation; this remains the six-class training MAT,
# never the consumed locked-test artifact.
DEFAULT_TRAIN_MAT = r"E:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"
