# onepager.py - builds the one-page PDF summary for sharing (funQ_night_clock_summary.pdf)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, ListFlowable, ListItem

ss = getSampleStyleSheet()
title = ParagraphStyle("t", parent=ss["Title"], fontSize=17, spaceAfter=2)
sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, textColor="#555555", alignment=1, spaceAfter=8)
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10, leading=13.5)
small = ParagraphStyle("sm", parent=ss["Normal"], fontSize=8, leading=10, textColor="#666666")

doc = SimpleDocTemplate("funQ_night_clock_summary.pdf", pagesize=letter,
                        leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.55 * inch, bottomMargin=0.5 * inch)
w = letter[0] - 1.4 * inch

def img(path, width):
    im = Image(path)
    im.drawHeight = im.drawHeight * width / im.drawWidth
    im.drawWidth = width
    return im

points = [
    "<b>Real data, no model:</b> Philadelphia and New Orleans look the same until 1am (~35% of bars open). "
    "Then Philadelphia drops to 3% at 2am, while New Orleans still has 14% open at 3am and 3% at 7am.",
    "<b>Inverse RL:</b> modeling each night as a sequence of choices, the model recovers what simulated groups value "
    "(cost, access, music, energy, going home) across 5 contexts, r = 0.85 to 0.99 with the true preferences, "
    "and beats a one-step choice model in every context.",
    "<b>Night clock:</b> later clocks keep groups chasing high-energy spots deeper into the night; "
    "in the simulation, Philadelphia groups are all home by 2am, the hour its real bars close.",
    "<b>Honest limit:</b> the bar cities can't reveal preferences for levers they don't have (covers, tables). "
    "What a city's institutions allow shapes what you can learn about the people in it.",
]

story = [
    Paragraph("funQ: How does a city's night clock change what groups value, and when?", title),
    Paragraph("Raúl Rojas &nbsp;|&nbsp; B.S. Computer Science + B.A. Anthropology, University of Florida", sub),
    Paragraph("Groups on a night out balance cost, access, music, friends and energy, and the balance shifts as the night goes on. "
              "This project infers those tradeoffs from behavior with maximum-entropy inverse reinforcement learning. "
              "It combines my fieldwork on Miami's nightlife scene with real venues, closing times and busyness from the "
              "Yelp Open Dataset (Philadelphia vs. New Orleans).", body),
    Spacer(1, 6),
    img("figures/4_real_closing_times.png", w * 0.78),
    Spacer(1, 4),
    img("figures/2_night_clock_headline.png", w),
    Spacer(1, 6),
    ListFlowable([ListItem(Paragraph(p, body), leftIndent=10) for p in points], bulletType="bullet", start="•", leftIndent=12),
    Spacer(1, 6),
    Paragraph("Groups are simulated (Yelp has no individual trajectories), so this is a proof of method, not a finding about real people. "
              "Simulated crowds track Yelp check-ins by venue and hour in Philadelphia (r = 0.70 / 0.74).", small),
    Paragraph("Venue data: Yelp Open Dataset, used under Yelp's terms for educational, non-commercial use.", small),
]
doc.build(story)
print("saved funQ_night_clock_summary.pdf")
