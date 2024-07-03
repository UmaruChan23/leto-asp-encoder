import pandas as pd
from enum import Enum, Flag, auto


class Encryption(Flag):
    YES = auto()
    NO = auto()
    PARTIALLY = auto()


class ApplicationInfo:
    def __init__(self, id: int, package: str, short_name: str, encryption: Encryption):
        self._id = id
        self._package = package
        self._short_name = short_name
        self._encryption = encryption

    def get_id(self) -> int:
        return self._id

    def get_package(self) -> str:
        return self._package

    def get_short_name(self) -> str:
        return self._short_name

    def get_encryption(self) -> Encryption:
        return self._encryption


class Application(Enum):
    F_RE = ApplicationInfo(id=13, short_name="F_RE", package="com.kirik88.fireader", encryption=Encryption.NO)
    GVL = ApplicationInfo(id=61, short_name="GVL", package="ru.godville.android", encryption=Encryption.NO)
    MK = ApplicationInfo(id=58, short_name="MK", package="com.mobilein.mk", encryption=Encryption.NO)
    NTV = ApplicationInfo(id=82, short_name="NTV", package="ru.ntv.client", encryption=Encryption.NO)
    PZ_EPR = ApplicationInfo(id=48, short_name="PZ_EPR", package="ru.itsilver.pizzaempire", encryption=Encryption.NO)
    WR = ApplicationInfo(id=23, short_name="WR", package="com.wolfram.android.alpha", encryption=Encryption.NO)
    HSN = ApplicationInfo(id=14, short_name="HSN", package="com.blizzard.wtcg.hearthstone", encryption=Encryption.YES)
    ISG = ApplicationInfo(id=3, short_name="ISG", package="com.instagram.android", encryption=Encryption.YES)
    MI_RU = ApplicationInfo(id=9, short_name="MI_RU", package="ru.mail.mailapp", encryption=Encryption.YES)
    PKB = ApplicationInfo(id=17, short_name="PKB", package="ru.pikabu.android", encryption=Encryption.YES)
    SB = ApplicationInfo(id=12, short_name="SB", package="ru.sberbankmobile", encryption=Encryption.YES)
    SP = ApplicationInfo(id=10, short_name="SP", package="com.skype.raider", encryption=Encryption.YES)
    FOUR_PDA = ApplicationInfo(id=2, short_name="4PDA", package="ru.fourpda.client", encryption=Encryption.PARTIALLY)
    BD = ApplicationInfo(id=18, short_name="BD", package="com.badoo.mobile", encryption=Encryption.PARTIALLY)
    BK = ApplicationInfo(id=46, short_name="BK", package="com.booking", encryption=Encryption.PARTIALLY)
    GG_CM = ApplicationInfo(id=5, short_name="GG_CM", package="com.android.chrome", encryption=Encryption.PARTIALLY)
    KMS = ApplicationInfo(id=11, short_name="KMS", package="com.nsadv.kommersant", encryption=Encryption.PARTIALLY)
    YD_BS = ApplicationInfo(id=57, short_name="YD_BS", package="com.yandex.browser", encryption=Encryption.PARTIALLY)


APPLICATIONS_MAP = {
    item.value.get_id(): item for item in Application
}


class Feature(Enum):
    APP_ID = "app_id"
    L3_TOT_PL_SZ_C2S = "L3_Tot_Pl_Sz_C2S"
    L3_TOT_PL_SZ_S2C = "L3_Tot_Pl_Sz_S2C"
    L4_TOT_PL_SZ_C2S = "L4_Tot_Pl_Sz_C2S"
    L4_TOT_PL_SZ_S2C = "L4_Tot_Pl_Sz_S2C"
    L3_AVG_DTG_SZ_C2S = "L3_Avg_Dtg_Sz_C2S"
    L3_AVG_DTG_SZ_S2C = "L3_Avg_Dtg_Sz_S2C"
    L4_AVG_PL_SZ_C2S = "L4_Avg_Pl_Sz_C2S"
    L4_AVG_PL_SZ_S2C = "L4_Avg_Pl_Sz_S2C"
    L3_STD_TOT_SZ_C2S = "L3_Std_Tot_Sz_C2S"
    L3_STD_TOT_SZ_S2C = "L3_Std_Tot_Sz_S2C"
    L4_STD_PL_SZ_C2S = "L4_Std_Pl_Sz_C2S"
    L4_STD_PL_SZ_S2C = "L4_Std_Pl_Sz_S2C"
    L3_AVG_PAC4MSG_C2S = "L3_Avg_Pac4Msg_C2S"
    L3_AVG_PAC4MSG_S2C = "L3_Avg_Pac4Msg_S2C"
    L3_EFFICIENCY_C2S = "L3_Efficiency_C2S"
    L3_EFFICIENCY_S2C = "L3_Efficiency_S2C"
    L3_TOT_DTG_SZ_CS_RATIO = "L3_Tot_Dtg_Sz_CS_ratio"
    L4_TOT_PL_SZ_CS_RATIO = "L4_Tot_Pl_Sz_CS_ratio"
    L3_TOT_DTG_CNT_CS_RATIO = "L3_Tot_Dtg_Cnt_CS_ratio"
    L3_TOT_DTG_CNT_C2S = "L3_Tot_Dtg_Cnt_C2S"
    L3_TOT_DTG_CNT_S2C = "L3_Tot_Dtg_Cnt_S2C"
    L3_SRC_ADDR = "L3_Src_Addr"
    L3_DST_ADDR = "L3_Dst_Addr"


FEATURES_MAP = {
    item.value: item for item in Feature
}


def load_data(
        encryption: Encryption = Encryption.YES | Encryption.NO | Encryption.PARTIALLY,
        exclude_features: list[Feature] = (Feature.L3_SRC_ADDR, Feature.L3_DST_ADDR),
        exclude_application: list[Application] = tuple()) -> pd.DataFrame:
    df = pd.read_csv("data/android_apps_traffic_attributes_prepared.csv")
    condition = None
    if Encryption.YES in encryption:
        condition = df["app_encryption"] == "yes" if condition is None else condition | (df["app_encryption"] == "yes")
    if Encryption.NO in encryption:
        condition = df["app_encryption"] == "no" if condition is None else condition | (df["app_encryption"] == "no")
    if Encryption.PARTIALLY in encryption:
        condition = df["app_encryption"] == "partially" if condition is None else condition | (df["app_encryption"] == "partially")

    result: pd.DataFrame = df[condition]
    result = result.drop(
        [
            'flow_id',
            'app_encryption',
            'app_package_name'
        ] + [item.value for item in exclude_features],
        axis=1)
    return result[~result["app_id"].isin([app.value.get_id() for app in exclude_application])]


def print_data_info(data: pd.DataFrame, title: str, show_features: bool = False):
    print("*"*32)
    print(title)
    print()
    print(f"Total instances: {len(data)}")
    print()
    print("Applications:")
    for app_id, count in data[Feature.APP_ID.value].value_counts().items():
        app_info = APPLICATIONS_MAP[app_id].value
        print(
            f"{app_info.get_id()}: {app_info.get_short_name()} ({app_info.get_package()}), encryption: {app_info.get_encryption().name} - {count} instance(s)")
    features = [feature for feature in data.columns if feature in FEATURES_MAP]
    print()
    print(f"Total features: {len(features)}")
    if show_features:
        print()
        print("Features:")
        for feature in features:
            print(feature)
    print("*"*32)
