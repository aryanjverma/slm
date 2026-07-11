#!/usr/bin/env python3
"""Extract a structured APUSH knowledge base from AMSCO 2016 (searchable PDF).

Deterministic: same PDF + same chapter map → same JSONL/index outputs.
pdf_index (0-based) = book_page + 37.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pdfplumber

PDF_PAGE_OFFSET = 37  # book_page + 37 = 0-based pdf index
SOFT_HYPHEN = "\u00ad"

# (chapter, book_page_start, title, period, date_range)
CHAPTERS: list[tuple[int, int, str, int, str]] = [
    (1, 2, "A New World of Many Cultures, 1491-1607", 1, "1491-1607"),
    (2, 24, "The Thirteen Colonies and the British Empire, 1607-1754", 2, "1607-1754"),
    (3, 45, "Colonial Society in the 18th Century", 2, "1700-1775"),
    (4, 69, "Imperial Wars and Colonial Protest, 1754-1774", 3, "1754-1774"),
    (5, 85, "The American Revolution and Confederation, 1774-1787", 3, "1774-1787"),
    (6, 103, "The Constitution and the New Republic, 1787-1800", 3, "1787-1800"),
    (7, 131, "The Age of Jefferson, 1800-1816", 4, "1800-1816"),
    (8, 150, "Nationalism and Economic Development, 1816-1848", 4, "1816-1848"),
    (9, 173, "Sectionalism, 1820-1860", 4, "1820-1860"),
    (10, 191, "The Age of Jackson, 1824-1844", 4, "1824-1844"),
    (11, 207, "Society, Culture, and Reform, 1820-1860", 4, "1820-1860"),
    (12, 230, "Territorial and Economic Expansion, 1830-1860", 5, "1830-1860"),
    (13, 247, "The Union in Peril, 1848-1861", 5, "1848-1861"),
    (14, 268, "The Civil War, 1861-1865", 5, "1861-1865"),
    (15, 291, "Reconstruction, 1863-1877", 5, "1863-1877"),
    (16, 319, "The Rise of Industrial America, 1865-1900", 6, "1865-1900"),
    (17, 339, "The Last West and the New South, 1865-1900", 6, "1865-1900"),
    (18, 360, "The Growth of Cities and American Culture, 1865-1900", 6, "1865-1900"),
    (19, 380, "The Politics of the Gilded Age, 1877-1900", 6, "1877-1900"),
    (20, 409, "Becoming a World Power, 1865-1917", 7, "1865-1917"),
    (21, 431, "The Progressive Era, 1901-1917", 7, "1901-1917"),
    (22, 454, "World War I and Its Aftermath, 1914-1920", 7, "1914-1920"),
    (23, 475, "The Modern Era of the 1920s", 7, "1920-1929"),
    (24, 496, "The Great Depression and the New Deal, 1929-1939", 7, "1929-1939"),
    (25, 521, "Diplomacy and World War II, 1929-1945", 7, "1929-1945"),
    (26, 557, "Truman and the Cold War, 1945-1952", 8, "1945-1952"),
    (27, 579, "The Eisenhower Years, 1952-1960", 8, "1952-1960"),
    (28, 600, "Promise and Turmoil, The 1960s", 8, "1960-1969"),
    (29, 625, "Limits of a Superpower, 1969-1980", 8, "1969-1980"),
    (30, 654, "Conservative Resurgence, 1980-2000", 9, "1980-2000"),
    (31, 679, "Challenges of the 21st Century", 9, "2000-present"),
]

# Typical APUSH misconceptions / oversimplifications by chapter (deterministic fallbacks).
CHAPTER_MISCONCEPTIONS: dict[int, list[str]] = {
    1: [
        "Columbus 'discovered' an empty continent; Native societies were sparse and primitive.",
        "All Native American cultures were nomadic hunter-gatherers.",
        "The Columbian Exchange only transferred plants and animals, not disease or people.",
        "Spanish colonization was uniquely cruel while other Europeans were benevolent.",
    ],
    2: [
        "All thirteen colonies were founded for religious freedom.",
        "Colonial slavery began fully formed in 1619 and never changed.",
        "New England, Middle, and Southern colonies were culturally identical.",
        "Native peoples quickly disappeared after first contact.",
    ],
    3: [
        "Colonial America was already a democracy with equal rights for all.",
        "The Great Awakening only mattered for religion, not politics or identity.",
        "Enlightenment ideas had little influence on ordinary colonists.",
        "Women and free Blacks had the same legal status as white male property owners.",
    ],
    4: [
        "The American Revolution began solely because of taxes on tea.",
        "All colonists united immediately against Britain after 1763.",
        "The French and Indian War had little connection to later imperial conflict.",
        "Protest was purely about abstract liberty, not economic interest.",
    ],
    5: [
        "Every American supported independence from the start.",
        "The Revolution instantly abolished slavery and inequality.",
        "The Articles of Confederation were a total failure with no achievements.",
        "Military victory alone created a stable national government.",
    ],
    6: [
        "The Constitution was universally popular and quickly ratified without conflict.",
        "Federalists and Anti-Federalists disagreed only about a Bill of Rights.",
        "The new government immediately resolved all sectional and fiscal disputes.",
        "Washington's presidency was nonpartisan and free of political conflict.",
    ],
    7: [
        "Jefferson consistently followed a strict construction of the Constitution.",
        "The Louisiana Purchase was uncontroversial and clearly constitutional.",
        "The War of 1812 was a decisive U.S. military triumph.",
        "Federalist opposition ended cleanly with no lasting effects.",
    ],
    8: [
        "The Era of Good Feelings meant there was no political conflict.",
        "The Market Revolution benefited all Americans equally.",
        "Henry Clay's American System was fully enacted without opposition.",
        "Nationalism erased sectional economic differences.",
    ],
    9: [
        "Sectionalism was only about slavery and nothing else.",
        "The North was uniformly industrial and the South uniformly plantation.",
        "Immigration had little effect on antebellum politics or cities.",
        "Free Blacks in the North enjoyed full civil equality.",
    ],
    10: [
        "Jacksonian Democracy extended political equality to all Americans.",
        "The Bank War was a simple fight of people versus elites with no economic costs.",
        "Indian Removal was inevitable and uncontested.",
        "Nullification proved states could permanently veto federal law.",
    ],
    11: [
        "Antebellum reform movements were unified and agreed on goals.",
        "Abolitionism was the majority Northern position before the Civil War.",
        "Women's rights advanced smoothly alongside other reforms.",
        "Transcendentalism had no connection to social reform.",
    ],
    12: [
        "Manifest Destiny was an uncontested national consensus.",
        "Westward expansion was mainly about empty land, not empire or slavery.",
        "The Mexican-American War was a minor border dispute.",
        "All migrants west were independent yeoman farmers.",
    ],
    13: [
        "The Civil War was inevitable from 1820 onward.",
        "Compromise always failed and never delayed conflict.",
        "Bleeding Kansas and Dred Scott were isolated events without connection.",
        "Lincoln's election alone caused secession with no deeper causes.",
    ],
    14: [
        "The Civil War was fought only to free enslaved people from day one.",
        "The Union had every advantage and victory was never in doubt.",
        "Emancipation immediately ended racial inequality.",
        "The Confederacy lost solely because of lack of will, not resources or strategy.",
    ],
    15: [
        "Reconstruction was a complete failure with no lasting gains.",
        "The Freedmen's Bureau accomplished nothing.",
        "Sharecropping was free-labor equality in practice.",
        "The Compromise of 1877 had no connection to the collapse of Black political rights.",
    ],
    16: [
        "Industrialization lifted all workers equally into the middle class.",
        "Robber barons and captains of industry are mutually exclusive labels.",
        "Labor unions were illegal and had no successes in this era.",
        "Laissez-faire meant the government never aided business.",
    ],
    17: [
        "The West was an empty frontier settled only by rugged individuals.",
        "The New South successfully industrialized like the North.",
        "Native resistance ended cleanly after one battle.",
        "Jim Crow was a sudden postwar invention unrelated to Redeemer politics.",
    ],
    18: [
        "All immigrants quickly and willingly assimilated.",
        "Urban machines only harmed cities and never provided services.",
        "Nativism was a fringe attitude with no policy impact.",
        "City culture was purely highbrow and ignored working-class life.",
    ],
    19: [
        "Gilded Age politics were meaningless theater with no real stakes.",
        "Both parties agreed on every economic issue.",
        "Populists were only rural cranks with no lasting influence.",
        "Civil service reform ended all patronage overnight.",
    ],
    20: [
        "U.S. imperialism was purely idealistic and never economic.",
        "The Spanish-American War was unwanted and accidental.",
        "Anti-imperialists had no serious arguments or political strength.",
        "Open Door policy meant China was treated as an equal partner.",
    ],
    21: [
        "Progressives agreed on a single coherent program.",
        "Progressivism ended corruption completely.",
        "Women's suffrage was granted solely because of World War I.",
        "Jim Crow and segregation were outside Progressive concern or complicity.",
    ],
    22: [
        "The U.S. entered World War I only to make the world safe for democracy.",
        "Wilson's Fourteen Points were fully adopted at Versailles.",
        "The home front saw no repression of dissent.",
        "Rejection of the League was only about isolationist stubbornness.",
    ],
    23: [
        "The 1920s were uniformly prosperous and carefree for all Americans.",
        "Prohibition was widely obeyed.",
        "Cultural conflict (Scopes, Klan, immigration restriction) was minor.",
        "Republican policy caused the boom with no structural weaknesses.",
    ],
    24: [
        "The Depression was caused only by the stock market crash.",
        "The New Deal ended the Depression by itself.",
        "All New Deal programs were constitutionally uncontroversial.",
        "Hoover did absolutely nothing in response to the crisis.",
    ],
    25: [
        "The U.S. was fully isolationist until Pearl Harbor with no prior involvement.",
        "Appeasement uniquely caused World War II with no other factors.",
        "The atomic bombs were the only reason Japan surrendered.",
        "The wartime alliance with the Soviet Union had no postwar tensions built in.",
    ],
    26: [
        "The Cold War was caused solely by Soviet aggression (or solely by U.S. aggression).",
        "Containment was applied the same way everywhere.",
        "McCarthyism had no earlier roots in Truman-era loyalty programs.",
        "The Korean War was a minor police action without lasting effects.",
    ],
    27: [
        "The 1950s were a silent, conformist decade with no dissent.",
        "Eisenhower ended the Cold War's military buildup.",
        "Brown v. Board immediately desegregated all schools.",
        "Suburbia was equally open to all racial and economic groups.",
    ],
    28: [
        "The Civil Rights Movement was only Martin Luther King Jr. and nonviolence.",
        "The Great Society ended poverty.",
        "Vietnam escalation was forced on presidents with no choices.",
        "1960s protest was only about college students and culture, not policy.",
    ],
    29: [
        "Watergate was only about a burglary, not broader executive power.",
        "Detente permanently ended Cold War rivalry.",
        "The energy crisis and stagflation had purely domestic causes.",
        "After Vietnam, the U.S. abandoned all foreign interventions.",
    ],
    30: [
        "Reaganomics had only benefits and no tradeoffs.",
        "The Conservative Resurgence ended liberalism completely.",
        "The Cold War ended solely because of U.S. military spending.",
        "Culture-war issues replaced all economic conflict in the 1980s–1990s.",
    ],
    31: [
        "Globalization only created winners in the U.S. economy.",
        "September 11 permanently unified American politics.",
        "Partisan polarization began only in the 21st century.",
        "Technology and the internet reduced social and political conflict.",
    ],
}

# High-value LEQ evidence names grounded in each AMSCO chapter (fill/ensure 8–15).
CHAPTER_EVIDENCE: dict[int, list[str]] = {
    1: [
        "Columbian Exchange",
        "Treaty of Tordesillas (1494)",
        "encomienda system",
        "asiento system",
        "Valladolid Debate",
        "New Laws of 1542",
        "Bartolome de Las Casas",
        "Hernan Cortes",
        "Francisco Pizarro",
        "Roanoke Island",
        "Iroquois Confederation",
        "Juan Gines de Sepulveda",
    ],
    2: [
        "Jamestown (1607)",
        "Mayflower Compact",
        "House of Burgesses",
        "Act of Toleration (Maryland)",
        "Bacon's Rebellion",
        "King Philip's War",
        "Navigation Acts",
        "mercantilism",
        "triangular trade",
        "Halfway Covenant",
        "William Penn / Pennsylvania",
        "Dominion of New England",
    ],
    3: [
        "Great Awakening",
        "Jonathan Edwards",
        "George Whitefield",
        "Enlightenment",
        "John Peter Zenger trial",
        "Poor Richard's Almanack",
        "Phillis Wheatley",
        "colonial legislatures / town meetings",
        "established churches",
        "Scotch-Irish migration",
        "subsistence farming",
        "hereditary aristocracy (limited in colonies)",
    ],
    4: [
        "French and Indian War / Seven Years' War",
        "Albany Plan of Union",
        "Treaty of Paris (1763)",
        "Proclamation of 1763",
        "Stamp Act (1765)",
        "Townshend Acts",
        "Boston Massacre",
        "Boston Tea Party",
        "Intolerable / Coercive Acts",
        "First Continental Congress",
        "Sons and Daughters of Liberty",
        "Committees of Correspondence",
    ],
    5: [
        "Second Continental Congress",
        "Declaration of Independence",
        "Battle of Saratoga",
        "Battle of Yorktown",
        "Treaty of Paris (1783)",
        "Articles of Confederation",
        "Northwest Ordinance (1787)",
        "Shays' Rebellion",
        "Common Sense (Thomas Paine)",
        "Loyalists / Tories",
        "Valley Forge",
        "Franco-American Alliance",
    ],
    6: [
        "Constitutional Convention (1787)",
        "Great Compromise",
        "Three-Fifths Compromise",
        "Federalist Papers",
        "Bill of Rights",
        "Judiciary Act of 1789",
        "Hamilton's financial plan",
        "Whiskey Rebellion",
        "Washington's Farewell Address",
        "Alien and Sedition Acts",
        "Kentucky and Virginia Resolutions",
        "XYZ Affair",
    ],
    7: [
        "Election of 1800",
        "Louisiana Purchase",
        "Marbury v. Madison",
        "Embargo Act (1807)",
        "War of 1812",
        "Battle of New Orleans",
        "Hartford Convention",
        "Lewis and Clark Expedition",
        "Chesapeake-Leopard affair",
        "Nonintercourse Act",
        "Treaty of Ghent",
        "Barbary pirates conflict",
    ],
    8: [
        "American System (Henry Clay)",
        "Era of Good Feelings",
        "Panic of 1819",
        "Missouri Compromise (1820)",
        "Monroe Doctrine",
        "McCulloch v. Maryland",
        "Gibbons v. Ogden",
        "Adams-Onis Treaty",
        "Tariff of 1816",
        "Erie Canal",
        "Market Revolution",
        "Dartmouth College v. Woodward",
    ],
    9: [
        "cotton gin / King Cotton",
        "peculiar institution",
        "Nat Turner's rebellion",
        "planter aristocracy",
        "Irish and German immigration",
        "nativism / Know-Nothings",
        "urbanization in the Northeast",
        "Old Northwest agriculture",
        "free African Americans in the North",
        "mountain whites",
        "Deep South plantation belt",
        "sectional economic specialization",
    ],
    10: [
        "universal white male suffrage",
        "spoils system",
        "Nullification Crisis",
        "Tariff of Abominations",
        "Indian Removal Act (1830)",
        "Worcester v. Georgia",
        "Trail of Tears",
        "Bank War / Second Bank of the U.S.",
        "Specie Circular",
        "Panic of 1837",
        "Whig Party",
        "Log Cabin campaign (1840)",
    ],
    11: [
        "Second Great Awakening",
        "temperance movement",
        "Seneca Falls Convention (1848)",
        "Declaration of Sentiments",
        "abolitionism / William Lloyd Garrison",
        "Frederick Douglass",
        "transcendentalism",
        "utopian communities (Brook Farm, Oneida)",
        "public school movement (Horace Mann)",
        "Dorothea Dix / asylum reform",
        "cult of domesticity",
        "American Colonization Society",
    ],
    12: [
        "Manifest Destiny",
        "Texas Revolution / Alamo",
        "Oregon Trail",
        "Mexican-American War",
        "Treaty of Guadalupe Hidalgo",
        "Mexican Cession",
        "Wilmot Proviso",
        "California Gold Rush",
        "Gadsden Purchase",
        "Ostend Manifesto",
        "Clayton-Bulwer Treaty",
        "Mormon migration",
    ],
    13: [
        "Compromise of 1850",
        "Fugitive Slave Act",
        "Uncle Tom's Cabin",
        "Kansas-Nebraska Act",
        "Bleeding Kansas",
        "Dred Scott v. Sandford",
        "Lincoln-Douglas debates",
        "John Brown's raid on Harpers Ferry",
        "Election of 1860",
        "secession of the Lower South",
        "Crittenden Compromise",
        "Free-Soil Party",
    ],
    14: [
        "Fort Sumter",
        "Anaconda Plan",
        "Emancipation Proclamation",
        "Gettysburg",
        "Vicksburg",
        "Sherman's March to the Sea",
        "Appomattox Court House",
        "Homestead Act (1862)",
        "Morrill Land Grant Act",
        "Pacific Railway Act",
        "New York City draft riots",
        "Thirteenth Amendment",
    ],
    15: [
        "Freedmen's Bureau",
        "Black Codes",
        "Civil Rights Act of 1866",
        "Fourteenth Amendment",
        "Fifteenth Amendment",
        "Military Reconstruction Act (1867)",
        "impeachment of Andrew Johnson",
        "sharecropping / tenant farming",
        "Ku Klux Klan / Enforcement Acts",
        "Compromise of 1877",
        "scalawags and carpetbaggers",
        "Redeemers",
    ],
    16: [
        "transcontinental railroad",
        "vertical and horizontal integration",
        "Standard Oil / John D. Rockefeller",
        "Andrew Carnegie / Gospel of Wealth",
        "Sherman Antitrust Act (1890)",
        "Interstate Commerce Act",
        "Knights of Labor",
        "American Federation of Labor",
        "Haymarket Riot",
        "Homestead Strike",
        "Pullman Strike",
        "Social Darwinism",
    ],
    17: [
        "Homestead Act settlement",
        "Dawes Severalty Act (1887)",
        "Battle of Little Bighorn",
        "Wounded Knee Massacre",
        "Ghost Dance movement",
        "New South / Henry Grady",
        "Plessy v. Ferguson (1896)",
        "Jim Crow laws",
        "Booker T. Washington",
        "Ida B. Wells",
        "sharecropping in the New South",
        "cattle drives / open range",
    ],
    18: [
        "new immigration (Southern/Eastern Europe)",
        "Ellis Island / Angel Island",
        "political machines / Tammany Hall",
        "settlement houses / Jane Addams",
        "Social Gospel",
        "tenements / Jacob Riis",
        "Chinese Exclusion Act (1882)",
        "nativism",
        "skyscrapers / urban technology",
        "spectator sports / mass culture",
        "Ashcan School",
        "public high schools expansion",
    ],
    19: [
        "Gilded Age patronage politics",
        "Pendleton Civil Service Act",
        "Interstate Commerce Commission",
        "Sherman Silver Purchase Act",
        "Populist Party / Omaha Platform",
        "William Jennings Bryan Cross of Gold speech",
        "Election of 1896",
        "McKinley Tariff",
        "Panic of 1893",
        "Coxey's Army",
        "Solid South",
        "Mugwumps",
    ],
    20: [
        "Alfred Thayer Mahan",
        "Spanish-American War",
        "USS Maine",
        "Treaty of Paris (1898)",
        "Philippine-American War",
        "Open Door Policy",
        "Roosevelt Corollary",
        "Panama Canal",
        "Gentlemen's Agreement",
        "dollar diplomacy",
        "Anti-Imperialist League",
        "Platt Amendment",
    ],
    21: [
        "muckrakers",
        "Square Deal",
        "Pure Food and Drug Act",
        "Meat Inspection Act",
        "Clayton Antitrust Act",
        "Sixteenth Amendment",
        "Seventeenth Amendment",
        "Eighteenth Amendment",
        "Nineteenth Amendment",
        "Federal Reserve Act",
        "NAACP",
        "Triangle Shirtwaist fire",
    ],
    22: [
        "unrestricted submarine warfare",
        "Zimmermann Telegram",
        "Selective Service Act",
        "Espionage and Sedition Acts",
        "Schenck v. United States",
        "Fourteen Points",
        "Treaty of Versailles",
        "League of Nations debate",
        "Great Migration",
        "Red Scare / Palmer Raids",
        "War Industries Board",
        "Committee on Public Information",
    ],
    23: [
        "Return to Normalcy",
        "Teapot Dome scandal",
        "Scopes Trial",
        "Immigration Act of 1924",
        "Prohibition / Volstead Act",
        "Harlem Renaissance",
        "Jazz Age / flappers",
        "Sacco and Vanzetti",
        "Ku Klux Klan revival",
        "installment buying",
        "Ford assembly line",
        "Lost Generation",
    ],
    24: [
        "stock market crash (1929)",
        "Hawley-Smoot Tariff",
        "Bonus March",
        "Hundred Days / First New Deal",
        "CCC / WPA / AAA",
        "FDIC / Glass-Steagall Banking Act",
        "Social Security Act (1935)",
        "Wagner Act / NLRB",
        "Court-packing plan",
        "Dust Bowl",
        "Indian Reorganization Act",
        "Fair Labor Standards Act",
    ],
    25: [
        "Stimson Doctrine",
        "Neutrality Acts",
        "Lend-Lease Act",
        "Atlantic Charter",
        "Pearl Harbor",
        "D-Day / Normandy invasion",
        "island hopping",
        "Manhattan Project",
        "Yalta and Potsdam conferences",
        "Japanese American internment / Korematsu",
        "wartime mobilization / Rosie the Riveter",
        "United Nations founding",
    ],
    26: [
        "containment / George Kennan",
        "Truman Doctrine",
        "Marshall Plan",
        "Berlin Airlift",
        "NATO",
        "Korean War",
        "NSC-68",
        "Second Red Scare / McCarthyism",
        "Alger Hiss and Rosenberg cases",
        "Taft-Hartley Act",
        "Fair Deal",
        "GI Bill",
    ],
    27: [
        "modern Republicanism",
        "interstate highway system",
        "brinkmanship / massive retaliation",
        "CIA coups in Iran and Guatemala",
        "Sputnik / NASA",
        "Brown v. Board of Education",
        "Montgomery Bus Boycott",
        "Little Rock Nine",
        "Eisenhower Doctrine",
        "U-2 incident",
        "military-industrial complex",
        "Levittown / suburbanization",
    ],
    28: [
        "New Frontier",
        "Bay of Pigs / Cuban Missile Crisis",
        "March on Washington",
        "Civil Rights Act of 1964",
        "Voting Rights Act of 1965",
        "Great Society / War on Poverty",
        "Medicare and Medicaid",
        "Gulf of Tonkin Resolution",
        "Tet Offensive",
        "Students for a Democratic Society",
        "Betty Friedan / NOW",
        "assassinations of MLK and RFK",
    ],
    29: [
        "Vietnamization",
        "Kent State shootings",
        "Pentagon Papers",
        "detente / SALT I",
        "Nixon visit to China",
        "Watergate scandal",
        "resignation of Nixon",
        "oil embargo / stagflation",
        "Camp David Accords",
        "Iran hostage crisis",
        "Roe v. Wade",
        "Environmental Protection Agency",
    ],
    30: [
        "Reaganomics / supply-side economics",
        "Strategic Defense Initiative",
        "Iran-Contra affair",
        "fall of the Berlin Wall",
        "Persian Gulf War (1991)",
        "Contract with America",
        "NAFTA",
        "Clinton impeachment",
        "Welfare Reform Act (1996)",
        "Moral Majority / Religious Right",
        "AIDS crisis",
        "1990s technology boom",
    ],
    31: [
        "Bush v. Gore (2000)",
        "September 11 attacks",
        "War in Afghanistan",
        "Iraq War (2003)",
        "Patriot Act / Department of Homeland Security",
        "Great Recession (2008)",
        "Affordable Care Act",
        "Obama election (2008)",
        "Tea Party movement",
        "Citizens United v. FEC",
        "same-sex marriage legalization",
        "social media and polarization",
    ],
}

YEAR_RE = re.compile(r"\b((?:1[4-9]\d{2}|20[0-2]\d)(?:\s*[-–]\s*(?:1[4-9]\d{2}|20[0-2]\d))?)\b")
EVIDENCE_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,6}\s+"
    r"(?:Act|Acts|Treaty|Treaties|Compromise|Proclamation|Doctrine|Amendment|"
    r"Amendments|Plan|Purchase|Revolution|War|Wars|Battle|Battles|Movement|"
    r"Convention|Code|System|Tariff|Bill|Laws?|Agreement|Accords?|Crisis|"
    r"Rebellion|Revolt|Massacre|Affair|Note|Address|Corollary|Policy)"
    r"(?:\s+of\s+\d{4})?"
    r")\b"
)
CASE_RE = re.compile(
    r"\b([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3}\s+v\.?\s+"
    r"[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3})\b"
)
SKIP_SENT_RE = re.compile(
    r"(which of the following|refer to the excerpt|multiple[\s\-]*choice|"
    r"think as a historian|long[\s\-]*essay|document[\s\-]*based|"
    r"learning objective|this chapter will|as you read|"
    r"use complete sentences|write a|answer the|see page|"
    r"preparing for the advanced placement|key terms by theme|"
    r"questions?\s+\d+|historical perspectives:|"
    r"civilians employed by the federal|year post office|source:\s*u\.s|"
    r"it is enough to make the whole world)",
    re.I,
)
CHAPTER_TITLE_PREFIX_RE = re.compile(
    r"^(?:THE\s+)?[A-Z][A-Z0-9 ,:'’\-]{8,80},\s*\d{4}(?:\s*-\s*\d{4})?\s+"
)
HEADER_LINE_RE = re.compile(
    r"^(?:"
    r"U\.S\.\s*HISTORY:.*|"
    r"PERIOD\s+\d+.*|"
    r"[A-Z0-9 ,:'’\-]{12,90}\d{1,3}"
    r")\s*$"
)
SECTION_CUT_RE = re.compile(
    r"(KEY TERMS BY THEME|MULTIPLE[\s\-]*CHOICE QUESTIONS|"
    r"SHORT[\s\-]*ANSWER QUESTIONS|THINK AS A HISTORIAN|"
    r"Questions\s+\d+\s*[-–]\s*\d+\s+refer)",
    re.I,
)
HP_RE = re.compile(r"HISTORICAL PERSPECTIVES\s*:\s*(.+)$", re.I | re.S)
STOP_EVIDENCE = {
    "Civil War",
    "Cold War",
    "World War",
    "New Deal",
    "Great Depression",
    "American Revolution",
    "United States",
    "Native Americans",
    "African Americans",
}


def book_page_end(chapter_num: int) -> int:
    """Last book page to scan for a chapter (exclusive of next chapter start)."""
    idx = next(i for i, c in enumerate(CHAPTERS) if c[0] == chapter_num)
    if idx + 1 < len(CHAPTERS):
        return CHAPTERS[idx + 1][1] - 1
    return 700  # through Think As a Historian / end of ch 31 content


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Soft hyphen may sit mid-word or at a line break: colo­nies / Amer­\nica
    text = re.sub(SOFT_HYPHEN + r"\n?", "", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("®", "")
    return text


def clean_page_text(raw: str) -> str:
    """Remove soft hyphens and rejoin end-of-line hyphenation; keep newlines."""
    text = normalize_unicode(raw)
    # Explicit hyphenation across line break: colo-\nnies -> colonies
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    # Drop obvious running headers / footers
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if HEADER_LINE_RE.match(stripped):
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        # Skip chart/table residue lines
        if re.search(r"\bSource:\s*U\.S\.", stripped, re.I):
            continue
        if re.fullmatch(r"[\d\s,.\-–]+", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def collapse_ws(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    # Fix common OCR/spacing artifacts after hyphen cleanup
    text = re.sub(r"\b([a-z]{2,})([A-Z][a-z]+)\b", r"\1 \2", text)
    return text.strip()


def extract_raw_chapter_pages(pdf: pdfplumber.PDF, book_start: int, book_end: int) -> str:
    chunks: list[str] = []
    for book_page in range(book_start, book_end + 1):
        pdf_index = book_page + PDF_PAGE_OFFSET
        if pdf_index < 0 or pdf_index >= len(pdf.pages):
            break
        page_text = pdf.pages[pdf_index].extract_text() or ""
        chunks.append(clean_page_text(page_text))
    return "\n".join(chunks)


def split_body_and_hp(raw: str) -> tuple[str, str, str]:
    """Return (body, historical_perspectives, key_terms_block)."""
    key_terms = ""
    kt = re.search(r"KEY TERMS BY THEME(.+?)(?=MULTIPLE|SHORT[\s\-]*ANSWER|THINK AS|Questions\s+\d+|\Z)", raw, re.I | re.S)
    if kt:
        key_terms = kt.group(1)
    cut = SECTION_CUT_RE.search(raw)
    truncated = raw[: cut.start()] if cut else raw
    hp_match = re.search(r"HISTORICAL PERSPECTIVES\s*:", truncated, re.I)
    if hp_match:
        body = truncated[: hp_match.start()]
        hp = truncated[hp_match.start() :]
    else:
        body = truncated
        hp = ""
    return body, hp, key_terms


def split_sentences(text: str) -> list[str]:
    text = collapse_ws(text)
    # Drop leading chapter title / epigraph noise roughly by finding first long prose
    parts = re.split(r"(?<=[.!?])\s+(?=[\"A-Z])", text)
    out: list[str] = []
    for part in parts:
        s = part.strip(" \t\"'")
        s = re.sub(r"\s+", " ", s)
        if s:
            out.append(s)
    return out


def tidy_fact(sentence: str) -> str:
    s = sentence.strip()
    s = CHAPTER_TITLE_PREFIX_RE.sub("", s)
    # "Hawley-Smoot Tariff (1930) In June..." -> drop leading label
    s = re.sub(
        r"^[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*){0,6}"
        r"(?:\s*\([^)]*\))?\s+"
        r"(?=(?:The|In|On|By|After|Before|During|Although|While|When|Under|"
        r"Between|From|At|For|As|This|These|Those|His|Her|Their|It|An|A|"
        r"Congress|President|Lincoln|Jefferson|Washington|Roosevelt)\b)",
        "",
        s,
        count=1,
    )
    # Strip leading epigraph attribution crumbs like "Sherman, June 30, 1864 The Civil..."
    s = re.sub(
        r"^[A-Z][a-zA-Z.'’\-]+(?:,\s+[A-Z][a-zA-Z.'’\-]+)*,\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},\s+\d{4}\s+",
        "",
        s,
    )
    # "John Crevecoeur, Letters from an American Farmer, 1782 The Frenchman..."
    s = re.sub(
        r"^[A-Z][^.]{0,80},\s+\d{4}\s+(?=[A-Z])",
        "",
        s,
        count=1,
    )
    # "Freedmen in the War After the Emancipation..." — drop short lead-in before After/In
    s = re.sub(
        r"^[A-Z][^.!?]{0,40}?\b(?=After the |In (?:January|February|March|April|May|June|July|August|September|October|November|December) )",
        "",
        s,
        count=1,
    )
    # "Roosevelt, said in his 1933 inaugural..." — drop broken attribution
    s = re.sub(r"^[A-Z][a-zA-Z.'’\-]+,\s+said in\b", "He said in", s, count=1)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!\"":
        s += "."
    return s


def score_fact(sentence: str) -> int:
    if SKIP_SENT_RE.search(sentence):
        return -100
    if "?" in sentence:
        return -100
    n = len(sentence)
    if n < 55 or n > 300:
        return -100
    # Reject table-like / numeric-heavy residue
    digit_ratio = sum(ch.isdigit() for ch in sentence) / max(n, 1)
    if digit_ratio > 0.18:
        return -100
    score = 0
    years = YEAR_RE.findall(sentence)
    score += min(len(years), 3) * 4
    if EVIDENCE_RE.search(sentence) or CASE_RE.search(sentence):
        score += 5
    if re.search(
        r"\b(signed|passed|established|created|declared|ratified|defeated|founded|"
        r"enacted|abolished|elected|appointed|invaded|annexed|emancipat|prohibited|"
        r"vetoed|overturned|negotiated|surrendered|assassinated|impeached|seceded)\w*\b",
        sentence,
        re.I,
    ):
        score += 4
    if re.search(
        r"\b(Congress|President|Constitution|Supreme Court|Parliament|colony|colonies|"
        r"slavery|enslaved|treaty|amendment|federal|republican|democrat)\b",
        sentence,
        re.I,
    ):
        score += 2
    # Prefer concrete over rhetorical
    if sentence.lower().startswith(("thus ", "therefore ", "however ", "indeed ")):
        score -= 1
    if re.search(r"\b(some historians|historians disagree|according to this interpretation)\b", sentence, re.I):
        score -= 2
    return score


def select_facts(body: str, min_n: int = 25, max_n: int = 40) -> list[str]:
    scored: list[tuple[int, str]] = []
    for sent in split_sentences(body):
        cleaned = tidy_fact(sent)
        sc = score_fact(cleaned)
        if sc >= 4:
            scored.append((sc, cleaned))
    scored.sort(key=lambda x: (-x[0], x[1]))
    facts: list[str] = []
    seen: set[str] = set()
    for sc, sent in scored:
        key = re.sub(r"\W+", "", sent.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        facts.append(sent)
        if len(facts) >= max_n:
            break
    if len(facts) < min_n:
        # Second pass: accept weaker but still informative sentences
        for sent in split_sentences(body):
            cleaned = tidy_fact(sent)
            sc = score_fact(cleaned)
            if sc < 1:
                continue
            key = re.sub(r"\W+", "", cleaned.lower())[:80]
            if key in seen:
                continue
            if not YEAR_RE.search(cleaned) and not EVIDENCE_RE.search(cleaned):
                # require at least one proper-name-ish token
                if not re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", cleaned):
                    continue
            seen.add(key)
            facts.append(cleaned)
            if len(facts) >= min_n:
                break
    return facts[:max_n]


def _normalize_evidence_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,;.")
    # Drop duplicated phrases: "Cold War Cold War"
    parts = name.split()
    half = len(parts) // 2
    if len(parts) >= 4 and half > 0 and parts[:half] == parts[half : 2 * half]:
        name = " ".join(parts[:half])
    # Reject obvious OCR glue (lowercase then Capital mid-token)
    if re.search(r"[a-z][A-Z]", name):
        return ""
    return name


def _is_quality_evidence(name: str) -> bool:
    if not name or len(name) < 5 or len(name) > 90:
        return False
    low = name.lower().strip()
    if low in {s.lower() for s in STOP_EVIDENCE}:
        return False
    if re.match(r"^(?:the|during|after|before|under|into|from|with)\b", low):
        return False
    if re.match(r"^(?:english|french|spanish)\s+policy\b", low):
        return False
    if re.search(r"\bPolicy\b", name) and not re.search(
        r"\b(Open Door|Good Neighbor|dollar diplomacy)\b", name, re.I
    ):
        # Allow curated Open Door etc.; reject bare "X Policy" extractions
        if name.endswith("Policy") and len(name.split()) <= 2:
            return False
    if re.search(r"[a-z][A-Z]", name):
        return False
    if name.count(" ") >= 8 and not YEAR_RE.search(name) and "v." not in name:
        return False
    return True


def merge_evidence(extracted: list[str], chapter: int, max_n: int = 15) -> list[str]:
    """Curated chapter evidence first, then high-quality extracted names."""
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = _normalize_evidence_name(name)
        if not _is_quality_evidence(name):
            return
        if re.fullmatch(r"Act of \d{4}", name):
            return
        if re.fullmatch(r"(?:New Laws|Railway Act|Monetary System)", name, re.I):
            return
        key = name.lower()
        # Skip if a longer existing entry already covers this name
        for existing in seen:
            if key != existing and (key in existing or existing in key):
                if len(key) < len(existing):
                    return
        # Drop shorter entries already stored
        drop = [e for e in seen if e != key and (e in key)]
        if drop:
            keep = []
            for item in out:
                low = item.lower()
                if low in drop:
                    seen.discard(low)
                    continue
                keep.append(item)
            out[:] = keep
        if key in seen:
            return
        seen.add(key)
        out.append(name)

    for name in CHAPTER_EVIDENCE.get(chapter, []):
        add(name)
        if len(out) >= max_n:
            return out[:max_n]
    for name in extracted:
        add(name)
        if len(out) >= max_n:
            break
    return out[:max_n]


def extract_evidence(
    body: str,
    key_terms: str,
    facts: list[str] | None = None,
    max_n: int = 15,
) -> list[str]:
    candidates: list[str] = []
    search_blobs = [body, key_terms]
    if facts:
        search_blobs.append(" ".join(facts))

    for blob in search_blobs:
        for match in EVIDENCE_RE.finditer(blob):
            candidates.append(match.group(1).strip())
        for match in CASE_RE.finditer(blob):
            candidates.append(match.group(1).strip())

    for token in re.findall(
        r"\b([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,5}\s+"
        r"(?:Act|Treaty|Compromise|Proclamation|Doctrine|System|Plan|Purchase|War|"
        r"Battle|Movement|Rebellion|Tariff|Bill|Code|Laws?)"
        r"(?:\s+of\s+\d{4})?)\b",
        key_terms,
    ):
        candidates.append(token.strip())

    extra_blob = body if not facts else body + "\n" + "\n".join(facts)
    for match in re.finditer(
        r"\b((?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|"
        r"Eleventh|Twelfth|Thirteenth|Fourteenth|Fifteenth|Sixteenth|Seventeenth|"
        r"Eighteenth|Nineteenth|Twentieth|Twenty-first)\s+Amendment)\b",
        extra_blob,
    ):
        candidates.append(match.group(1))
    for match in re.finditer(
        r"\b((?:[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+){0,4}\s+)?"
        r"(?:Treaty|Panic|Election|Compromise|Revolution|War|Battle|Act|Crisis)"
        r"\s+of\s+\d{4})\b",
        extra_blob,
    ):
        phrase = match.group(1)
        if re.fullmatch(r"Act of \d{4}", phrase):
            continue
        candidates.append(phrase)

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name in candidates:
        name = _normalize_evidence_name(name)
        if not _is_quality_evidence(name):
            continue
        if not re.search(
            r"\b(Act|Treaty|Compromise|Proclamation|Doctrine|Amendment|v\.|War|Battle|"
            r"Purchase|Tariff|Bill|System|Plan|Movement|Rebellion|Massacre|Affair|"
            r"Convention|Code|Laws?|Crisis|Note|Address|Corollary|Accords?|"
            r"Panic|Election)\b",
            name,
        ):
            continue
        key = name.lower()
        if key in seen:
            continue
        bonus = 0
        if re.search(r"\b(Act|Treaty|Compromise|Proclamation|Doctrine|Amendment|v\.)\b", name):
            bonus += 4
        if YEAR_RE.search(name):
            bonus += 3
        if re.search(r"\b(War|Battle|Purchase|Movement|Rebellion|Panic|Election)\b", name):
            bonus += 2
        seen.add(key)
        scored.append((bonus, name))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:max_n]]


def misconceptions_from_hp(hp_text: str, chapter: int) -> list[str]:
    """Distill opposing interpretations from Historical Perspectives + curated list."""
    curated = list(CHAPTER_MISCONCEPTIONS.get(chapter, []))
    distilled: list[str] = []
    if hp_text:
        hp_body = collapse_ws(hp_text)
        hp_body = re.sub(r"^HISTORICAL PERSPECTIVES\s*:\s*", "", hp_body, flags=re.I)
        title_m = re.match(r"([^?]+\?)", hp_body)
        if title_m:
            q = title_m.group(1).strip()
            q = re.sub(r"\s+", " ", q)
            # Title case cleanup
            if q.isupper():
                q = q.title().replace("A ", "a ").replace("The ", "the ").replace("Of ", "of ")
                q = q[0].upper() + q[1:] if q else q
            distilled.append(
                f"Treating one answer to '{q}' as settled fact instead of a live historical debate."
            )
        # Contrastive claims framed as student pitfalls
        for sent in split_sentences(hp_body):
            if not re.search(
                r"\b(argue[ds]?|revisionist|traditionally|critics|apologists|"
                r"historians|interpretation|disagree|portray|blame|praise|"
                r"orthodox|view)\b",
                sent,
                re.I,
            ):
                continue
            cleaned = tidy_fact(sent)
            if not (70 <= len(cleaned) <= 240):
                continue
            distilled.append(
                f"Students may oversimplify by accepting only one side: {cleaned}"
            )
            if len(distilled) >= 3:
                break

    merged: list[str] = []
    seen: set[str] = set()
    for item in distilled + curated:
        key = item.lower()[:70]
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= 6:
            break
    if len(merged) < 3:
        for item in curated:
            if item not in merged:
                merged.append(item)
            if len(merged) >= 3:
                break
    return merged[:6]


def context_hooks_from_body(body: str, date_range: str, min_n: int = 4, max_n: int = 8) -> list[str]:
    """Broader contextualization sentences useful before the chapter timeframe."""
    sents = split_sentences(body)
    hooks: list[str] = []
    seen: set[str] = set()
    # Prefer early sentences that set scene / prior causes
    for sent in sents[:50]:
        cleaned = tidy_fact(sent)
        if SKIP_SENT_RE.search(cleaned) or "?" in cleaned:
            continue
        if not (55 <= len(cleaned) <= 280):
            continue
        if score_fact(cleaned) < 0 and not YEAR_RE.search(cleaned):
            continue
        if re.search(
            r"\b(before|prior|after|following|led to|resulting|background|"
            r"by the (early|late|mid)|during the|in the (wake|aftermath)|"
            r"inherited|long[- ]term|context|since|from the)\b",
            cleaned,
            re.I,
        ) or YEAR_RE.search(cleaned):
            key = cleaned.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            hooks.append(cleaned)
        if len(hooks) >= max_n:
            break
    if len(hooks) < min_n:
        for sent in sents[:30]:
            cleaned = tidy_fact(sent)
            if not (70 <= len(cleaned) <= 260):
                continue
            if SKIP_SENT_RE.search(cleaned) or "?" in cleaned:
                continue
            if score_fact(cleaned) < 0:
                continue
            key = cleaned.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            hooks.append(cleaned)
            if len(hooks) >= min_n:
                break
    frame = (
        f"Developments in the years leading into {date_range} shaped the political, "
        f"economic, and social conflicts covered in this chapter."
    )
    if len(hooks) < min_n:
        hooks.append(frame)
    return hooks[:max_n]


def topic_keywords(title: str, facts: list[str], evidence: list[str]) -> list[str]:
    words: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", title):
        if token.lower() not in {"with", "from", "that", "this", "century", "american", "america"}:
            words.append(token.lower())
    for name in evidence[:12]:
        words.append(name.lower())
    # Frequent content words from facts
    freq: dict[str, int] = defaultdict(int)
    stop = {
        "that", "this", "with", "from", "were", "was", "have", "had", "their", "they",
        "which", "when", "into", "also", "been", "more", "than", "only", "over",
        "after", "before", "between", "under", "would", "could", "about", "other",
        "these", "such", "many", "most", "some", "what", "while", "where", "during",
        "united", "states", "american", "america", "people", "government",
    }
    for fact in facts:
        for w in re.findall(r"[A-Za-z][a-z]{3,}", fact.lower()):
            if w not in stop:
                freq[w] += 1
    for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        words.append(w)
    # unique preserve order
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:40]


def extract_chapter(
    pdf: pdfplumber.PDF,
    chapter: int,
    book_start: int,
    title: str,
    period: int,
    date_range: str,
) -> dict[str, Any]:
    book_end = book_page_end(chapter)
    raw = extract_raw_chapter_pages(pdf, book_start, book_end)
    body, hp, key_terms = split_body_and_hp(raw)
    facts = select_facts(body)
    extracted_evidence = extract_evidence(body, key_terms, facts=facts)
    evidence = merge_evidence(extracted_evidence, chapter)
    misconceptions = misconceptions_from_hp(hp, chapter)
    hooks = context_hooks_from_body(body, date_range)
    keywords = topic_keywords(title, facts, evidence)
    return {
        "id": f"amsco_ch{chapter:02d}",
        "chapter": chapter,
        "title": title,
        "period": period,
        "date_range": date_range,
        "book_page_start": book_start,
        "book_page_end": book_end,
        "key_facts": facts,
        "misconceptions": misconceptions,
        "evidence_bank": evidence,
        "context_hooks": hooks,
        "topic_keywords": keywords,
        "source": "AMSCO United States History: Preparing for the Advanced Placement Examination (2016)",
    }


def build_index(chapters: list[dict[str, Any]]) -> dict[str, Any]:
    by_period: dict[str, list[str]] = defaultdict(list)
    keyword_to_chapters: dict[str, list[str]] = defaultdict(list)
    for ch in chapters:
        cid = ch["id"]
        by_period[str(ch["period"])].append(cid)
        for kw in ch.get("topic_keywords", []):
            keyword_to_chapters[kw].append(cid)
    return {
        "source": "amsco_2016",
        "chapter_count": len(chapters),
        "periods": {p: by_period[p] for p in sorted(by_period, key=int)},
        "chapters": [
            {
                "id": ch["id"],
                "chapter": ch["chapter"],
                "title": ch["title"],
                "period": ch["period"],
                "date_range": ch["date_range"],
                "book_page_start": ch["book_page_start"],
                "fact_count": len(ch["key_facts"]),
                "topic_keywords": ch.get("topic_keywords", [])[:20],
            }
            for ch in chapters
        ],
        "topic_keywords": {
            kw: ids for kw, ids in sorted(keyword_to_chapters.items(), key=lambda kv: kv[0])
            if len(kw) >= 4
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AMSCO 2016 APUSH knowledge base.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("amsco_2016_searchable.pdf"),
        help="Path to searchable AMSCO 2016 PDF",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/knowledge"),
        help="Directory for kb jsonl and index",
    )
    parser.add_argument(
        "--chapters",
        type=str,
        default="",
        help="Optional comma-separated chapter numbers (default: all)",
    )
    args = parser.parse_args()

    wanted: set[int] | None = None
    if args.chapters.strip():
        wanted = {int(x.strip()) for x in args.chapters.split(",") if x.strip()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "amsco_2016_kb.jsonl"
    index_path = args.output_dir / "amsco_2016_kb_index.json"

    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    with pdfplumber.open(str(args.pdf)) as pdf:
        print(f"Opened {args.pdf} ({len(pdf.pages)} pages)")
        for chapter, book_start, title, period, date_range in CHAPTERS:
            if wanted is not None and chapter not in wanted:
                continue
            try:
                row = extract_chapter(pdf, chapter, book_start, title, period, date_range)
                n_facts = len(row["key_facts"])
                if n_facts < 15:
                    failed.append(
                        {
                            "chapter": chapter,
                            "reason": f"low_fact_count:{n_facts}",
                            "title": title,
                        }
                    )
                results.append(row)
                print(
                    f"Ch {chapter:02d} P{period}: {n_facts} facts, "
                    f"{len(row['evidence_bank'])} evidence, "
                    f"{len(row['misconceptions'])} misconceptions, "
                    f"{len(row['context_hooks'])} hooks"
                )
            except Exception as exc:  # noqa: BLE001 — report per-chapter failure
                failed.append({"chapter": chapter, "reason": f"error:{exc}", "title": title})
                print(f"Ch {chapter:02d} FAILED: {exc}")

    write_jsonl(jsonl_path, results)
    index = build_index(results)
    index["failed_chapters"] = failed
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_facts = sum(len(r["key_facts"]) for r in results)
    print(f"Wrote {len(results)} chapters → {jsonl_path}")
    print(f"Wrote index → {index_path}")
    print(f"Total facts: {total_facts}")
    if failed:
        print(f"Chapters with issues: {failed}")


if __name__ == "__main__":
    main()
