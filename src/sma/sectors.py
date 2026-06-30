"""Static GICS sector mapping for the universe.

Hand-curated from src/sma/universe.yaml's comment-based groupings. Used by
the backtest harness for the 25% sector cap risk rail. SPY is mapped to
its own "ETF" bucket so it doesn't consume sector budget for any equity sector.

When the universe changes, update this file. (Future: pull from yfinance
.info or a paid GICS feed.)
"""

# 2026-05-26: SPDR sector ETFs — the canonical benchmarks for each GICS
# sector. Used by rel_strength_sector_etf_30d to give each ticker a
# benchmark-relative momentum score (cleaner than the per-name peer-mean
# rel_strength_sector_30d, which depends on universe composition: a
# small sector with 2 peers gives a noisier mean than one with 20).
SECTOR_ETF_FOR_GICS: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}


def sector_etf_for(ticker: str) -> str | None:
    """Return the SPDR sector ETF ticker that benchmarks `ticker`'s GICS
    sector, or None if the ticker is itself an ETF / unmapped sector."""
    sec = SECTORS.get(ticker)
    if sec is None:
        return None
    return SECTOR_ETF_FOR_GICS.get(sec)


SECTORS: dict[str, str] = {
    # ETFs / benchmarks
    "SPY": "ETF",
    "NANC": "ETF",  # politician-trades thematic ETF (added 2026-05-09)
    # 2026-05-26: SPDR sector ETFs added for rel_strength_sector_etf_30d
    # feature. Bucketed "Sector ETF" so they're tracked + ingested but don't
    # consume any equity sector's budget. They WILL get model predictions
    # alongside individual names — they almost never rank top-K because
    # index-level returns are typically below the best single names, but
    # the model treats them as valid training samples.
    "XLB": "Sector ETF",
    "XLC": "Sector ETF",
    "XLE": "Sector ETF",
    "XLF": "Sector ETF",
    "XLI": "Sector ETF",
    "XLK": "Sector ETF",
    "XLP": "Sector ETF",
    "XLRE": "Sector ETF",
    "XLU": "Sector ETF",
    "XLV": "Sector ETF",
    "XLY": "Sector ETF",

    # Universe expansion 2026-05-10 — big-cap names previously omitted.
    "TSM": "Information Technology",
    "ASML": "Information Technology",
    "AMAT": "Information Technology",
    "LRCX": "Information Technology",
    "PYPL": "Information Technology",
    "PLTR": "Information Technology",
    "SNOW": "Information Technology",
    "CRWD": "Information Technology",
    "DDOG": "Information Technology",
    "SHOP": "Information Technology",
    "ARM": "Information Technology",
    "COIN": "Financials",
    "BRK.B": "Financials",
    "ABNB": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "UBER": "Industrials",
    "TMUS": "Communication Services",

    # AI-sphere expansion 2026-05-11 (semi/optical/networking).
    "MU": "Information Technology",
    "MRVL": "Information Technology",
    "KLAC": "Information Technology",
    "MCHP": "Information Technology",
    "ON": "Information Technology",
    "LITE": "Information Technology",
    "ANET": "Information Technology",
    "MPWR": "Information Technology",

    # Cross-sector broader sweep 2026-05-11.
    "WBD": "Communication Services",
    "ROKU": "Communication Services",
    "CMG": "Consumer Discretionary",
    "LULU": "Consumer Discretionary",
    "ULTA": "Consumer Discretionary",
    "MAR": "Consumer Discretionary",
    "HLT": "Consumer Discretionary",
    "DASH": "Consumer Discretionary",
    "MDLZ": "Consumer Staples",
    "TGT": "Consumer Staples",
    "EOG": "Energy",
    "OXY": "Energy",
    "MPC": "Energy",
    "USB": "Financials",
    "PNC": "Financials",
    "ICE": "Financials",
    "CME": "Financials",
    "BX": "Financials",
    "KKR": "Financials",
    "MDT": "Health Care",
    "ISRG": "Health Care",
    "ELV": "Health Care",
    "REGN": "Health Care",
    "VRTX": "Health Care",
    "BSX": "Health Care",
    "GD": "Industrials",
    "FDX": "Industrials",
    "UNP": "Industrials",
    "NOC": "Industrials",
    "ADI": "Information Technology",
    "NXPI": "Information Technology",
    "FTNT": "Information Technology",
    "PANW": "Information Technology",
    "NET": "Information Technology",
    "ECL": "Materials",
    "SHW": "Materials",
    "SO": "Utilities",
    "AEP": "Utilities",

    # Information Technology (was "Tech mega-cap" in universe.yaml)
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOGL": "Information Technology",
    "AMZN": "Information Technology",
    "META": "Information Technology",
    "NVDA": "Information Technology",
    "TSLA": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "CSCO": "Information Technology",
    "ACN": "Information Technology",
    "INTC": "Information Technology",
    "AMD": "Information Technology",
    "QCOM": "Information Technology",
    "TXN": "Information Technology",
    "IBM": "Information Technology",
    "INTU": "Information Technology",
    "NOW": "Information Technology",

    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    "SCHW": "Financials",
    "BLK": "Financials",
    "AXP": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "SPGI": "Financials",

    # Health Care (GICS spelling, was "Health Care" in universe.yaml)
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "PFE": "Health Care",
    "MRK": "Health Care",
    "ABBV": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "DHR": "Health Care",
    "BMY": "Health Care",
    "AMGN": "Health Care",
    "GILD": "Health Care",
    "CVS": "Health Care",

    # Consumer Discretionary
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",

    # Consumer Staples
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "PM": "Consumer Staples",
    "MO": "Consumer Staples",

    # Industrials
    "CAT": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "BA": "Industrials",
    "DE": "Industrials",
    "GE": "Industrials",
    "LMT": "Industrials",
    "RTX": "Industrials",

    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",

    # Materials (split from "Materials / utilities" in universe.yaml)
    "LIN": "Materials",
    "APD": "Materials",

    # Utilities (split from "Materials / utilities" in universe.yaml)
    "NEE": "Utilities",
    "DUK": "Utilities",

    # Communication Services (GICS name since 2018, was "Comms / media" in universe.yaml)
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
    "CMCSA": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",

    # Universe expansion 2026-05-24 — fills sector holes (Real Estate was
    # absent entirely; Insurance was a Financials gap; clean energy, EV,
    # biotech, REIT-as-AI-infra were single-name or zero coverage).
    # Real Estate (NEW sector; cap-ratio rail's 25% applies per-sector so
    # adding a sector dilutes existing concentration, doesn't tighten it).
    "EQIX": "Real Estate",
    "DLR": "Real Estate",
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "O": "Real Estate",
    "VICI": "Real Estate",
    # Insurance (within Financials)
    "PGR": "Financials",
    "CB": "Financials",
    "MET": "Financials",
    "AIG": "Financials",
    # Fintech
    "SOFI": "Financials",
    "HOOD": "Financials",
    "AFRM": "Financials",
    # Gaming + interactive
    "DKNG": "Consumer Discretionary",
    "EA": "Communication Services",     # GICS classes game publishers as comms
    "TTWO": "Communication Services",
    "RBLX": "Communication Services",
    # Clean energy (Industrials per GICS for these names)
    "ENPH": "Information Technology",   # GICS classes solar-inverter as IT
    "FSLR": "Information Technology",
    # Metals + mining
    "NUE": "Materials",
    "FCX": "Materials",
    "NEM": "Materials",
    # EV (Consumer Discretionary per GICS for auto-OEMs)
    "RIVN": "Consumer Discretionary",
    # Biotech
    "MRNA": "Health Care",
    # Software
    "WDAY": "Information Technology",
    # Crypto-leveraged equity proxy
    "MSTR": "Information Technology",
    # Industrials specialty
    "EMR": "Industrials",
    "ETN": "Industrials",
    # Consumer
    "BBY": "Consumer Discretionary",
    "MNST": "Consumer Staples",
    # Comms / streaming
    "SPOT": "Communication Services",
    # 2026-06-12 universe expansion (+81 names)
    "ITW": "Industrials",
    "PH": "Industrials",
    "CMI": "Industrials",
    "ROK": "Industrials",
    "CSX": "Industrials",
    "NSC": "Industrials",
    "DAL": "Industrials",
    "UAL": "Industrials",
    "LUV": "Industrials",
    "WM": "Industrials",
    "RSG": "Industrials",
    "PCAR": "Industrials",
    "CARR": "Industrials",
    "OTIS": "Industrials",
    "JCI": "Industrials",
    "FAST": "Industrials",
    "URI": "Industrials",
    "PWR": "Industrials",
    "AME": "Industrials",
    "CPRT": "Industrials",
    "ODFL": "Industrials",
    "PSX": "Energy",
    "VLO": "Energy",
    "WMB": "Energy",
    "KMI": "Energy",
    "HAL": "Energy",
    "DVN": "Energy",
    "FANG": "Energy",
    "BKR": "Energy",
    "CL": "Consumer Staples",
    "KMB": "Consumer Staples",
    "GIS": "Consumer Staples",
    "SYY": "Consumer Staples",
    "KR": "Consumer Staples",
    "STZ": "Consumer Staples",
    "HSY": "Consumer Staples",
    "KDP": "Consumer Staples",
    "CHD": "Consumer Staples",
    "DG": "Consumer Staples",
    "D": "Utilities",
    "EXC": "Utilities",
    "SRE": "Utilities",
    "XEL": "Utilities",
    "ED": "Utilities",
    "WEC": "Utilities",
    "PEG": "Utilities",
    "ES": "Utilities",
    "DOW": "Materials",
    "DD": "Materials",
    "VMC": "Materials",
    "MLM": "Materials",
    "ALB": "Materials",
    "CCI": "Real Estate",
    "PSA": "Real Estate",
    "SPG": "Real Estate",
    "WELL": "Real Estate",
    "AVB": "Real Estate",
    "CI": "Health Care",
    "HUM": "Health Care",
    "MCK": "Health Care",
    "SYK": "Health Care",
    "BDX": "Health Care",
    "ZTS": "Health Care",
    "BRK-B": "Financials",
    "AON": "Financials",
    "TFC": "Financials",
    "ALL": "Financials",
    "AFL": "Financials",
    "TRV": "Financials",
    "CHTR": "Communication Services",
    "OMC": "Communication Services",
    "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary",
    "ROST": "Consumer Discretionary",
    "YUM": "Consumer Discretionary",
    "DHI": "Consumer Discretionary",
    "LEN": "Consumer Discretionary",
    "EBAY": "Consumer Discretionary",
    "DPZ": "Consumer Discretionary",
}


def sector_for(ticker: str) -> str:
    """Return the GICS sector for a ticker. Defaults to 'Unknown' if missing."""
    return SECTORS.get(ticker, "Unknown")


def sector_map_for(universe: list[str]) -> dict[str, str]:
    """Return a sector map for the given universe (compatibility with prior stub interface)."""
    return {t: sector_for(t) for t in universe}
