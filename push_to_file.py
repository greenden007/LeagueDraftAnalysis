import nn_preprocessing

st = """

    Aatrox

    Ahri

    Akali

    Akshan

    Alistar

    Ambessa

    Amumu

    Anivia

    Annie

    Aphelios

    Ashe

    Aurelion Sol

    Aurora

    Azir

    Bard

    Belveth

    Blitzcrank

    Brand

    Braum

    Briar

    Caitlyn

    Camille

    Cassiopeia

    Chogath

    Corki

    Darius

    Diana

    Dr. Mundo

    Draven

    Ekko

    Elise

    Evelynn

    Ezreal

    Fiddlesticks

    Fiora

    Fizz

    Galio

    Gangplank

    Garen

    Gnar

    Gragas

    Graves

    Gwen

    Hecarim

    Heimerdinger

    Hwei

    Illaoi

    Irelia

    Ivern

    Janna

    Jarvan IV

    Jax

    Jayce

    Jhin

    Jinx

    Kaisa

    Kalista

    Karma

    Karthus

    Kassadin

    Katarina

    Kayle

    Kayn

    Kennen

    KhaZix

    Kindred

    Kled

    KogMaw

    KSante

    LeBlanc

    Lee Sin

    Leona

    Lillia

    Lissandra

    Lucian

    Lulu

    Lux

    Malphite

    Malzahar

    Maokai

    Master Yi

    Mel

    Milio

    Miss Fortune

    Mordekaiser

    Morgana

    Naafiri

    Nami

    Nasus

    Nautilus

    Neeko

    Nidalee

    Nilah

    Nocturne

    Nunu & Willump

    Olaf

    Orianna

    Ornn

    Pantheon

    Poppy

    Pyke

    Qiyana

    Quinn

    Rakan

    Rammus

    Reksai

    Rell

    Renata Glasc

    Renekton

    Rengar

    Riven

    Rumble

    Ryze

    Samira

    Sejuani

    Senna

    Seraphine

    Sett

    Shaco

    Shen

    Shyvana

    Singed

    Sion

    Sivir

    Skarner

    Smolder

    Sona

    Soraka

    Swain

    Sylas

    Syndra

    Tahm Kench

    Taliyah

    Talon

    Taric

    Teemo

    Thresh

    Tristana

    Trundle

    Tryndamere

    Twisted Fate

    Twitch

    Udyr

    Urgot

    Varus

    Vayne

    Veigar

    VelKoz

    Vex

    Vi

    Viego

    Viktor

    Vladimir

    Volibear

    Warwick

    Wukong

    Xayah

    Xerath

    Xin Zhao

    Yasuo

    Yone

    Yorick

    Yuumi

    Zac

    Zed

    Zeri

    Ziggs

    Zilean

    Zoe

    Zyra
"""

def champs_list():
    ls = []
    for line in st.splitlines():
        if line.strip():
            ls.append(line.strip())
    return ls
