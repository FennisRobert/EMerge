"""
╔══════════════════════════════════════════════════════════════════════╗
║  library.py – Material-property database                             ║
║                                                                      ║
║  This file is released under CC0 1.0 – Public-Domain Dedication.     ║
║  You may use, copy, modify, and distribute it without restriction.   ║
║                                                                      ║
║  All OTHER files in this repository are proprietary. See LICENSE     ║
║  at the project root for full terms, conditions, and disclaimers.    ║
║                                                                      ║
║  DISCLAIMER: Data herein are provided “as is”, with no warranties.   ║
║  Verify critical values independently before use.                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from ._emerge.material import Material, AIR, COPPER
from ._emerge.const import C0, Z0, PI, EPS0, MU0

EISO: float = (Z0/(2*PI))**0.5
EOMNI = (3*Z0/(4*PI))**0.5

##MATERIALS 
VACUUM = Material(color="#2d8cd5", opacity=0.05)

############################################################
#                         METALS                         #
############################################################

GREY = "#bfbfbf"
MET_ALUMINUM         = Material(cond=3.77e7,  color=GREY, opacity=0.5)
MET_CARBON           = Material(cond=3.33e4,  color=GREY, opacity=0.5)
MET_CHROMIUM         = Material(cond=5.56e6,  color=GREY, opacity=0.5)
MET_COPPER           = Material(cond=5.8e7,   color="#62290c", opacity=1.0)
MET_GOLD             = Material(cond=4.10e7,  color="#d4af37", opacity=0.5)
MET_INDIUM           = Material(cond=6.44e6,  color=GREY, opacity=0.5)
MET_IRIDIUM          = Material(cond=2.13e7,  color=GREY, opacity=0.5)
MET_IRON             = Material(cond=1.04e7,  color="#aaaaaa", opacity=0.5)
MET_LEAD             = Material(cond=4.84e6,  color=GREY, opacity=0.5)
MET_MAGNESIUM        = Material(cond=2.38e7,  color=GREY, opacity=0.5)
MET_NICKEL           = Material(cond=1.14e7,  color=GREY, opacity=0.5)
MET_NICHROME         = Material(cond=9.09e5,  color=GREY, opacity=0.5)
MET_PALLADIUM        = Material(cond=9.42e6,  color=GREY, opacity=0.5)
MET_PLATINUM         = Material(cond=9.42e6,  color=GREY, opacity=0.5)
MET_RHODIUM          = Material(cond=2.22e7,  color=GREY, opacity=0.5)
MET_SILVER           = Material(cond=6.29e7,  color=GREY, opacity=0.5)
MET_TANTALUM         = Material(cond=6.44e6,  color=GREY, opacity=0.5)
MET_TANTALUM_NITRIDE = Material(cond=3.97e5,  color=GREY, opacity=0.5)
MET_TIN              = Material(cond=8.66e6,  color=GREY, opacity=0.5)
MET_TITANIUM         = Material(cond=1.82e6,  color=GREY, opacity=0.5)
MET_TUNGSTEN         = Material(cond=1.79e7,  color=GREY, opacity=0.5)
MET_ZINC             = Material(cond=1.76e7,  color=GREY, opacity=0.5)
MET_ZIRCONIUM        = Material(cond=2.44e7,  color=GREY, opacity=0.5)


############################################################
#                       SEMICONDUCTORS                     #
############################################################

SEMI_SILICON       = Material(er=11.7,  tand=0.005,     color="#b4b4b4", opacity=0.5) #  Crystalline Si
SEMI_SILICON_N     = Material(er=7.5,   tand=0.0003,    color="#a0a0a0", opacity=0.5) #  Silicon Nitride (Si₃N₄)
SEMI_SILICON_OXIDE = Material(er=3.9,   tand=0.0001,    color="#e0e0e0", opacity=0.5) #  Silicon Dioxide (SiO₂)
SEMI_GERMANIUM     = Material(er=16.0,  tand=0.001,     color="#787878", opacity=0.5)
SEMI_GAAS          = Material(er=13.1,  tand=0.0016,    color="#aa8888", opacity=0.5) #  Gallium Arsenide
SEMI_GA_N          = Material(er=8.9,   tand=0.002,     color="#8888cc", opacity=0.5) #  Gallium Nitride
SEMI_INP           = Material(er=12.5,  tand=0.0015,    color="#cc99aa", opacity=0.5) #  Indium Phosphide
SEMI_ALN           = Material(er=8.6,   tand=0.0003,    color="#ccccee", opacity=0.5) #  Aluminum Nitride
SEMI_AL2O3         = Material(er=9.8,   tand=0.0002,    color="#eaeaea", opacity=0.5) #  Alumina
SEMI_SAPPHIRE      = Material(er=9.4,   tand=0.0001,    color="#ddddff", opacity=0.5)
SEMI_DIAMOND       = Material(er=5.5,   tand=0.00005,   color="#cceeff", opacity=0.5) #  Synthetic CVD diamond
SEMI_HBN           = Material(er=4.0,   tand=0.0001,    color="#eeeeff", opacity=0.5) #  Hexagonal Boron Nitride
SEMI_SIOXNY        = Material(er=5.0,   tand=0.002,     color="#ddddee", opacity=0.5) #  Silicon Oxynitride (SiOxNy)

############################################################
#                          LIQUIDS                         #
############################################################

LIQ_WATER     = Material(er=80.1,    cond=0.0,  color="#0080ff", opacity=0.3)
LIQ_FERRITE   = Material(er=12.0,    ur=2000, tand=0.02, color="#994d4d", opacity=0.3)

############################################################
#                        DIELECTRICS                       #
############################################################

#  TRADEMARKS: All product names (e.g. “DUROID”) are the property of their
#  respective owners.  Use of them here does not imply any affiliation with or
#  endorsement by those owners.

DIEL_PTFE           = Material(er=2.1, tand=0.0002, color="#21912b", opacity=0.3)
DIEL_POLYIMIDE      = Material(er=3.4, tand=0.02, color="#b8b8b8")
DIEL_CERAMIC        = Material(er=6.0, tand=0.001, color="#efead1")
DIEL_AD10           = Material(er=10.2, tand=0.0078, color="#21912b", opacity=0.3)
DIEL_AD1000         = Material(er=10.2, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_AD250          = Material(er=2.5, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_AD250_PIM      = Material(er=2.5, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_AD250A         = Material(er=2.50, tand=0.0015, color="#21912b", opacity=0.3)
DIEL_AD250C         = Material(er=2.50, tand=0.0014, color="#21912b", opacity=0.3)
DIEL_AD255          = Material(er=2.55, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_AD255A         = Material(er=2.55, tand=0.0015, color="#21912b", opacity=0.3)
DIEL_AD255C         = Material(er=2.55, tand=0.0014, color="#21912b", opacity=0.3)
DIEL_AD260A         = Material(er=2.60, tand=0.0017, color="#21912b", opacity=0.3)
DIEL_AD270          = Material(er=2.7, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_AD300          = Material(er=3, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD300_PIM      = Material(er=3, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD300A         = Material(er=3.00, tand=0.002, color="#21912b", opacity=0.3)
DIEL_AD300C         = Material(er=2.97, tand=0.002, color="#21912b", opacity=0.3)
DIEL_AD320          = Material(er=3.2, tand=0.0038, color="#21912b", opacity=0.3)
DIEL_AD320_PIM      = Material(er=3.2, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD320A         = Material(er=3.20, tand=0.0032, color="#21912b", opacity=0.3)
DIEL_AD350          = Material(er=3.5, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD350_PIM      = Material(er=3.5, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD350A         = Material(er=3.50, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD410          = Material(er=4.1, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD430          = Material(er=4.3, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD450          = Material(er=4.5, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_AD450A         = Material(er=4.5, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_AD5            = Material(er=5.1, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AD600          = Material(er=5.90, tand=0.003, color="#21912b", opacity=0.3)
DIEL_AR1000         = Material(er=9.8, tand=0.003, color="#21912b", opacity=0.3)
DIEL_CER_10         = Material(er=10.00, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_CLTE           = Material(er=2.96, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_CLTE_AT        = Material(er=3.00, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_CLTE_LC        = Material(er=2.94, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_CLTE_XT        = Material(er=2.94, tand=0.0012, color="#21912b", opacity=0.3)
DIEL_COMCLAD_HF_ER2 = Material(er=2, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_COMCLAD_HF_ER3 = Material(er=3, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_COMCLAD_HF_ER4 = Material(er=4, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_COMCLAD_HF_ER5 = Material(er=5, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_COMCLAD_HF_ER6 = Material(er=6, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_COPPER_CLAD_ULTEM = Material(er=3.05, tand=0.003, color="#21912b", opacity=0.3)
DIEL_CUCLAD_217LX   = Material(er=2.17, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_CUCLAD_233LX   = Material(er=2.33, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_CUCLAD_250GT   = Material(er=2.5, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_CUCLAD_250GX   = Material(er=2.4, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_CUFLON         = Material(er=2.05, tand=0.00045, color="#21912b", opacity=0.3)
DIEL_DICLAD_522     = Material(er=2.4, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_DICLAD_527     = Material(er=2.4, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_DICLAD_870     = Material(er=2.33, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_DICLAD_880     = Material(er=2.17, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_DICLAD_880_PIM = Material(er=2.17, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_GETEK          = Material(er=3.5, tand=0.01, color="#21912b", opacity=0.3)
DIEL_GETEK          = Material(er=3.8, tand=0.01, color="#21912b", opacity=0.3)
DIEL_IS6802_80      = Material(er=2.80, tand=0.003, color="#21912b", opacity=0.3)
DIEL_IS6803_00      = Material(er=3.00, tand=0.003, color="#21912b", opacity=0.3)
DIEL_IS6803_20      = Material(er=3.20, tand=0.003, color="#21912b", opacity=0.3)
DIEL_IS6803_33      = Material(er=3.33, tand=0.003, color="#21912b", opacity=0.3)
DIEL_IS6803_38      = Material(er=3.38, tand=0.0032, color="#21912b", opacity=0.3)
DIEL_IS6803_45      = Material(er=3.45, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_ISOCLAD_917    = Material(er=2.17, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_ISOCLAD_933    = Material(er=2.33, tand=0.0016, color="#21912b", opacity=0.3)
DIEL_ISOLA_I_TERA_MT = Material(er=3.45, tand=0.0030,      color="#3c9747")
DIEL_ISOLA_NELCO_4000_13 = Material(er=3.77, tand=0.008,  color="#3c9747")
DIEL_MAT_25N        = Material(er=3.38, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_MAT25FR        = Material(er=3.58, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_MEGTRON6R5775  = Material(er=3.61, tand=0.004, color="#21912b", opacity=0.3)
DIEL_MERCURYWAVE_9350 = Material(er=3.5, tand=0.004, color="#21912b", opacity=0.3)
DIEL_MULTICLAD_HF   = Material(er=3.7, tand=0.0045, color="#21912b", opacity=0.3)
DIEL_N_8000         = Material(er=3.5, tand=0.011, color="#21912b", opacity=0.3)
DIEL_N4350_13RF     = Material(er=3.5, tand=0.0065, color="#21912b", opacity=0.3)
DIEL_N4380_13RF     = Material(er=3.8, tand=0.007, color="#21912b", opacity=0.3)
DIEL_N8000Q         = Material(er=3.2, tand=0.006, color="#21912b", opacity=0.3)
DIEL_N9300_13RF     = Material(er=3, tand=0.004, color="#21912b", opacity=0.3)
DIEL_N9320_13RF     = Material(er=3.2, tand=0.0045, color="#21912b", opacity=0.3)
DIEL_N9338_13RF     = Material(er=3.38, tand=0.0046, color="#21912b", opacity=0.3)
DIEL_N9350_13RF     = Material(er=3.48, tand=0.0055, color="#21912b", opacity=0.3)
DIEL_NH9294         = Material(er=2.94, tand=0.0022, color="#21912b", opacity=0.3)
DIEL_NH9300         = Material(er=3.00, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_NH9320         = Material(er=3.20, tand=0.0024, color="#21912b", opacity=0.3)
DIEL_NH9338         = Material(er=3.38, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_NH9348         = Material(er=3.48, tand=0.003, color="#21912b", opacity=0.3)
DIEL_NH9350         = Material(er=3.50, tand=0.003, color="#21912b", opacity=0.3)
DIEL_NH9410         = Material(er=4.10, tand=0.003, color="#21912b", opacity=0.3)
DIEL_NH9450         = Material(er=4.50, tand=0.003, color="#21912b", opacity=0.3)
DIEL_NORCLAD        = Material(er=2.55, tand=0.0011, color="#21912b", opacity=0.3)
DIEL_NX9240         = Material(er=2.40, tand=0.0016, color="#21912b", opacity=0.3)
DIEL_NX9245         = Material(er=2.45, tand=0.0016, color="#21912b", opacity=0.3)
DIEL_NX9250         = Material(er=2.50, tand=0.0017, color="#21912b", opacity=0.3)
DIEL_NX9255         = Material(er=2.55, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_NX9260         = Material(er=2.60, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_NX9270         = Material(er=2.70, tand=0.002, color="#21912b", opacity=0.3)
DIEL_NX9294         = Material(er=2.94, tand=0.0022, color="#21912b", opacity=0.3)
DIEL_NX9300         = Material(er=3.00, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_NX9320         = Material(er=3.20, tand=0.0024, color="#21912b", opacity=0.3)
DIEL_NY9208         = Material(er=2.08, tand=0.0006, color="#21912b", opacity=0.3)
DIEL_NY9217         = Material(er=2.17, tand=0.0008, color="#21912b", opacity=0.3)
DIEL_NY9220         = Material(er=2.20, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_NY9233         = Material(er=2.33, tand=0.0011, color="#21912b", opacity=0.3)
DIEL_POLYGUIDE      = Material(er=2.320, tand=0.0005, color="#21912b", opacity=0.3)
DIEL_RF_30          = Material(er=3.00, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_RF_301         = Material(er=2.97, tand=0.0018, color="#21912b", opacity=0.3)
DIEL_RF_35          = Material(er=3.50, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_RF_35A2        = Material(er=3.50, tand=0.0015, color="#21912b", opacity=0.3)
DIEL_RF_35P         = Material(er=3.50, tand=0.0034, color="#21912b", opacity=0.3)
DIEL_RF_35TC        = Material(er=3.50, tand=0.0011, color="#21912b", opacity=0.3)
DIEL_RF_41          = Material(er=4.10, tand=0.0038, color="#21912b", opacity=0.3)
DIEL_RF_43          = Material(er=4.30, tand=0.0033, color="#21912b", opacity=0.3)
DIEL_RF_45          = Material(er=4.50, tand=0.0037, color="#21912b", opacity=0.3)
DIEL_RF_60A         = Material(er=6.15, tand=0.0038, color="#21912b", opacity=0.3)
DIEL_RO3003         = Material(er=3.00, tand=0.0011, color="#21912b", opacity=0.3)
DIEL_RO3006         = Material(er=6.15, tand=0.002, color="#21912b", opacity=0.3)
DIEL_RO3010         = Material(er=10.2, tand=0.0022, color="#21912b", opacity=0.3)
DIEL_RO3035         = Material(er=3.50, tand=0.0017, color="#21912b", opacity=0.3)
DIEL_RO3203         = Material(er=3.02, tand=0.0016, color="#21912b", opacity=0.3)
DIEL_RO3206         = Material(er=6.15, tand=0.0027, color="#21912b", opacity=0.3)
DIEL_RO3210         = Material(er=10.2, tand=0.0027, color="#21912b", opacity=0.3)
DIEL_RO3730         = Material(er=3.00, tand=0.0016, color="#21912b", opacity=0.3)
DIEL_RO4003C        = Material(er=3.38, tand=0.0029, color="#21912b", opacity=0.3)
DIEL_RO4350B        = Material(er=3.48, tand=0.0037, color="#21912b", opacity=0.3)
DIEL_RO4350B_TX     = Material(er=3.48, tand=0.0034, color="#21912b", opacity=0.3)
DIEL_RO4360         = Material(er=6.15, tand=0.0038, color="#21912b", opacity=0.3)
DIEL_RO4533         = Material(er=3.30, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_RO4534         = Material(er=3.40, tand=0.0027, color="#21912b", opacity=0.3)
DIEL_RO4535         = Material(er=3.50, tand=0.0037, color="#21912b", opacity=0.3)
DIEL_RO4730         = Material(er=3.00, tand=0.0033, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_5870 = Material(er=2.33, tand=0.0012, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_5880 = Material(er=2.20, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_5880LZ = Material(er=1.96, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_6002 = Material(er=2.94, tand=0.0012, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_6006 = Material(er=6.15, tand=0.0027, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_6010_2LM = Material(er=10.2, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_6035HTC = Material(er=3.50, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_6202 = Material(er=2.94, tand=0.0015, color="#21912b", opacity=0.3)
DIEL_RT_Duroid_6202PR = Material(er=2.90, tand=0.002, color="#21912b", opacity=0.3)
DIEL_SYRON_70000_002IN_Thick = Material(er=3.4, tand=0.0045, color="#21912b", opacity=0.3)
DIEL_SYRON_71000_004IN_THICK = Material(er=3.39, tand=0.005, color="#21912b", opacity=0.3)
DIEL_SYRON_71000INCH = Material(er=3.61, tand=0.006, color="#21912b", opacity=0.3)
DIEL_TACLAMPLUS= Material(er=2.10, tand=0.0004, color="#21912b", opacity=0.3)
DIEL_TC350          = Material(er=3.50, tand=0.002, color="#21912b", opacity=0.3)
DIEL_TC600          = Material(er=6.15, tand=0.002, color="#21912b", opacity=0.3)
DIEL_THETA          = Material(er=3.85, tand=0.0123, color="#21912b", opacity=0.3)
DIEL_TLA_6          = Material(er=2.62, tand=0.0017, color="#21912b", opacity=0.3)
DIEL_TLC_27         = Material(er=2.75, tand=0.003, color="#21912b", opacity=0.3)
DIEL_TLC_30         = Material(er=3.00, tand=0.003, color="#21912b", opacity=0.3)
DIEL_TLC_32         = Material(er=3.20, tand=0.003, color="#21912b", opacity=0.3)
DIEL_TLC_338        = Material(er=3.38, tand=0.0034, color="#21912b", opacity=0.3)
DIEL_TLC_35         = Material(er=3.50, tand=0.0037, color="#21912b", opacity=0.3)
DIEL_TLE_95         = Material(er=2.95, tand=0.0028, color="#21912b", opacity=0.3)
DIEL_TLF_34         = Material(er=3.40, tand=0.002, color="#21912b", opacity=0.3)
DIEL_TLF_35         = Material(er=3.50, tand=0.002, color="#21912b", opacity=0.3)
DIEL_TLG_29         = Material(er=2.87, tand=0.0027, color="#21912b", opacity=0.3)
DIEL_TLG_30         = Material(er=3,    tand=0.0038, color="#21912b", opacity=0.3)
DIEL_TLP_3          = Material(er=2.33, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TLP_5          = Material(er=2.20, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TLP_5A         = Material(er=2.17, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TLT_0          = Material(er=2.45, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLT_6          = Material(er=2.65, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLT_7          = Material(er=2.60, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLT_8          = Material(er=2.55, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLT_9          = Material(er=2.50, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLX_0          = Material(er=2.45, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLX_6          = Material(er=2.65, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLX_7          = Material(er=2.60, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLX_8          = Material(er=2.55, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLX_9          = Material(er=2.50, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_TLY_3          = Material(er=2.33, tand=0.0012, color="#21912b", opacity=0.3)
DIEL_TLY_3F         = Material(er=2.33, tand=0.0012, color="#21912b", opacity=0.3)
DIEL_TLY_5 = Material(er=2.20, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TLY_5_L = Material(er=2.20, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TLY_5A = Material(er=2.17, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TLY_5AL = Material(er=2.17, tand=0.0009, color="#21912b", opacity=0.3)
DIEL_TMM_10 = Material(er=9.20, tand=0.0022, color="#21912b", opacity=0.3)
DIEL_TMM_10i = Material(er=9.80, tand=0.002, color="#21912b", opacity=0.3)
DIEL_TMM_3 = Material(er=3.27, tand=0.002, color="#21912b", opacity=0.3)
DIEL_TMM_4 = Material(er=4.50, tand=0.002, color="#21912b", opacity=0.3)
DIEL_TMM_6 = Material(er=6.00, tand=0.0023, color="#21912b", opacity=0.3)
DIEL_TRF_41 = Material(er=4.10, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_TRF_43 = Material(er=4.30, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_TRF_45 = Material(er=4.50, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_TSM_26 = Material(er=2.60, tand=0.0014, color="#21912b", opacity=0.3)
DIEL_TSM_29 = Material(er=2.94, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_TSM_30 = Material(er=3.00, tand=0.0013, color="#21912b", opacity=0.3)
DIEL_TSM_DS = Material(er=2.85, tand=0.001, color="#21912b", opacity=0.3)
DIEL_TSM_DS3 = Material(er=3.00, tand=0.0011, color="#21912b", opacity=0.3)
DIEL_ULTRALAM_2000 = Material(er=2.4, tand=0.0019, color="#21912b", opacity=0.3)
DIEL_ULTRALAM_3850 = Material(er=2.9, tand=0.0025, color="#21912b", opacity=0.3)
DIEL_XT_Duroid_80000_002IN_Thick = Material(er=3.23, tand=0.0035, color="#21912b", opacity=0.3)
DIEL_XT_Duroid_8100 = Material(er=3.54, tand=0.0049, color="#21912b", opacity=0.3)
DIEL_XT_Duroid_81000_004IN_Thick = Material(er=3.32, tand=0.0038, color="#21912b", opacity=0.3)
DIEL_TEFLON = Material(er=2.1, tand=0.0003, color='#eeeeee', opacity=0.3)

# Legacy FR Materials
DIEL_FR1 = Material(er=4.8,  tand=0.025, color="#3c9747", opacity=0.3) # Paper + phenolic resin
DIEL_FR2 = Material(er=4.8,  tand=0.02,  color="#3c9747", opacity=0.3) # Paper + phenolic resin
DIEL_FR3 = Material(er=4.5,  tand=0.02,  color="#2b7a4b", opacity=0.3) # Paper + epoxy resin
DIEL_FR4 = Material(er=4.4,  tand=0.015, color="#1e8449", opacity=0.3) # Woven glass + epoxy resin (industry standard)
DIEL_FR5 = Material(er=4.2,  tand=0.012, color="#156e38", opacity=0.3) # Woven glass + high-temp epoxy resin
DIEL_FR6 = Material(er=5.2,  tand=0.030, color="#145a32", opacity=0.3) # Paper + unknown resin, poor thermal performance

# Magnetic Materials
MU_METAL  = Material(cond=1.0e6, ur=200000, color="#666680", opacity=0.3)


############################################################
#                           FOAMS                          #
############################################################

FOAM_ROHACELL_31  = Material(er=1.05,  tand=0.0005,  color="#f0e1a1", opacity=0.15) # PMI-based structural foam
FOAM_ROHACELL_51  = Material(er=1.07,  tand=0.0006,  color="#f0dea0", opacity=0.15) # denser version
FOAM_ROHACELL_71  = Material(er=1.10,  tand=0.0007,  color="#e5d199", opacity=0.15)
FOAM_PEI          = Material(er=1.15,  tand=0.0035,  color="#e0b56f", opacity=0.15) # polyetherimide-based foam
FOAM_PMI          = Material(er=1.10,  tand=0.0008,  color="#d9c690", opacity=0.15) # polymethacrylimide
FOAM_PVC          = Material(er=1.20,  tand=0.0040,  color="#cccccc", opacity=0.15)
FOAM_EPS          = Material(er=1.03,  tand=0.0050,  color="#f7f7f7", opacity=0.15) # expanded polystyrene
FOAM_XPS          = Material(er=1.05,  tand=0.0030,  color="#e0e0e0", opacity=0.15) # extruded polystyrene
FOAM_PU           = Material(er=1.10,  tand=0.0080,  color="#d0d0d0", opacity=0.15) # polyurethane foam
FOAM_GLAS         = Material(er=3.10,  tand=0.0050,  color="#888888", opacity=0.15) # cellular glass, denser
FOAM_AIREX_C70    = Material(er=1.10,  tand=0.0010,  color="#f7e7a3", opacity=0.15) # PET closed cell
FOAM_AIREX_T92    = Material(er=1.10,  tand=0.0020,  color="#f6d08a", opacity=0.15) # higher strength PET
FOAM_PVC_CORECELL = Material(er=1.56,  tand=0.0025,  color="#aaaaaa", opacity=0.15) # structural core PVC