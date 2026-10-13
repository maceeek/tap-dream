# tapdream 🌙

**Jedno dlouhé časosběrné video z kamery Tapo, které roste každou nocí.**

`tapdream` je CLI nástroj pro Linux napsaný v Rustu. Sám si stahuje záznamy ze SD karty kamery Tapo,
vybere z nich snímky a připojí je na konec **jednoho stále stejného souboru** `timelapse.mp4`.
Cílem je video, ve kterém je vidět, jak miminko roste a kolik nocí jste spolu strávili.

- **Soukromí na prvním místě.** Snímky s nahotou (typicky kojení) se automaticky vyřazují i s časovou rezervou
  kolem nich. Vše běží lokálně na vašem GPU, žádný obraz neopouští počítač.
- **Inteligentní tempo.** Když se něco děje, video běží pomaleji. Klidný spánek se zrychlí a prázdná postel
  (kamera zůstala zapnutá) se vynechá.
- **„Zajímavost“ snímků.** Model CLIP vybírá v každém časovém okně nejzajímavější a nejostřejší snímek a ukládá
  „fotku noci“.
- **Výkon.** Dekódování na GPU (NVDEC), dekódování jen klíčových snímků, inference na GPU (ONNX Runtime + CUDA),
  kódování na GPU (NVENC). Připojení nové noci zapíše jen nová data, nic se nepřegenerovává.
- **Robustní index.** SQLite eviduje, co se stáhlo, co se zpracovalo a co je ve výsledném videu. Přerušený běh
  (výpadek proudu) se při dalším spuštění sám dorovná.

```
 SD karta kamery ──► stažení (port 8800, AES) ──► index (SQLite) ──► zpracování noci
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┘
   ▼
 dekódování (NVDEC) ─► analýza snímků ─────────────────► výběr snímků ─────────► kódování (NVENC)
                       • pohyb, ostrost, tma             • adaptivní rychlost      │
                       • NudeNet (nahota)                • nejlepší snímek v okně  ▼
                       • CLIP (přítomnost, kojení,       • okno soukromí           segment noci (fMP4)
                         zajímavost)                       ±minuty kolem detekce   │
                                                                                   ▼
                                                     timelapse.mp4 ◄── připojení na konec (append)
```

> **Stav ověření.** Protokol kamery je přenesen z [pytapo](https://github.com/JurajNyiri/pytapo) (verze 3.4.26)
> a otestován proti referenčním vektorům vygenerovaným přímo z pytapo a proti simulátoru kamery
> (`tools/mock_camera.py`). Proti skutečné kameře zatím vyzkoušen nebyl. Pokud něco nepůjde, pomůže
> výstup `tapdream -v camera info`.

---

## Obsah

1. [Instalace](#instalace)
2. [První spuštění](#první-spuštění)
3. [Jak to funguje](#jak-to-funguje)
4. [Soukromí: vyřazování nahoty](#soukromí-vyřazování-nahoty)
5. [„Zajímavost“ snímků](#zajímavost-snímků)
6. [Jeden stále stejný soubor](#jeden-stále-stejný-soubor)
7. [Příkazy](#příkazy)
8. [Automatický běh (systemd)](#automatický-běh-systemd)
9. [Konfigurace – co ladit](#konfigurace--co-ladit)
10. [Výkon](#výkon)
11. [Řešení problémů](#řešení-problémů)
12. [Vývoj a testy](#vývoj-a-testy)
13. [Licence a poděkování](#licence-a-poděkování)

---

## Instalace

### 1. Systémové balíčky

```bash
# Debian/Ubuntu
sudo apt install ffmpeg curl build-essential pkg-config libssl-dev python3-venv
```

`ffmpeg` z distribuce obsahuje NVENC i NVDEC (`-hwaccel cuda`). Ke GPU je potřeba proprietární ovladač NVIDIA.

### 2. Sestavení

```bash
curl https://sh.rustup.rs -sSf | sh     # pokud ještě nemáte Rust
cargo build --release
sudo install -m 755 target/release/tapdream /usr/local/bin/
```

### 3. ONNX Runtime s podporou GPU

AI modely běží přes ONNX Runtime, který se načítá dynamicky. Nejjednodušší je instalace z pipu do
vlastního virtuálního prostředí, kde ji `tapdream` najde sám (a sám načte i knihovny CUDA/cuDNN z pip balíčků):

```bash
python3 -m venv ~/.tapdream-venv
~/.tapdream-venv/bin/pip install "onnxruntime-gpu[cuda,cudnn]"     # CUDA 13 (ovladač ≥ 580)
```

Se starším ovladačem (CUDA 12) použijte starší verzi:

```bash
~/.tapdream-venv/bin/pip install onnxruntime-gpu==1.22.0 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
```

Jinou instalaci lze zadat přes `gpu.onnxruntime_lib` v konfiguraci nebo proměnnou `ORT_DYLIB_PATH`.
Bez GPU vše funguje na CPU, jen pomaleji (`pip install onnxruntime`).

### 4. Modely

```bash
tapdream init                 # vytvoří config.toml
tapdream models fetch         # NudeNet 640m + 320n, CLIP ViT-B/32 (~650 MB) do data_dir/models
```

Modely se stahují jen jednou: NudeNet z GitHubu/PyPI, CLIP (OpenAI) z clip-as-service. To je jediný přístup
na internet. Veškerá analýza pak běží offline.

---

## První spuštění

1. Kamera musí mít **SD kartu** a zapnuté **nahrávání** (nepřetržité nebo při detekci pohybu) a musí být v LAN.
2. V `config.toml` vyplňte `camera.host` a **heslo k účtu TP-Link/Tapo** (cloudový účet, ke kterému je
   kamera přidaná). Kamera ho vyžaduje pro přístup k záznamům, komunikace ale probíhá jen lokálně.
   Heslo můžete předat i bezpečněji: proměnnou `TAPDREAM_CLOUD_PASSWORD` nebo souborem (`cloud_password_file`).
3. Nastavte `general.timezone`, `schedule.capture_hours` (např. `"18:00-10:00"`) a `overlay.birth_date`.
4. Zkontrolujte prostředí:

```bash
tapdream doctor --camera
```

5. Spusťte první běh:

```bash
tapdream run          # stáhne nové záznamy a zpracuje dokončené noci
tapdream status
```

> **Pozor na zámek kamery:** po několika chybných přihlášeních kamera dočasně blokuje přihlašování.
> `tapdream` proto při špatném hesle hned skončí a neopakuje pokusy.

---

## Jak to funguje

### Noci

Den je rozdělený na „noci“: **noc D trvá od `day_boundary` (12:00) dne D do 12:00 dne D+1**. Do noci se počítají
jen záběry z `capture_hours`. Noc se zpracuje, až skončí (+ `grace_minutes`) a jsou staženy všechny její záznamy.
Z každé noci vznikne jeden segment, který se připojí na konec časosběru. Do obrazu se vypálí titulek, např.
`Noc 102 · 10. 10. 2026 · 3 měsíce 9 dní` (šablona `overlay.template`, počítadlo nocí od `birth_date`).

### Index

`data_dir/index.db` (SQLite) eviduje:

| tabulka | obsah |
|---|---|
| `recordings` | každý záznam na kameře: časy, stav (`new` → `downloaded` → `purged`, `skipped`, `failed`), počet pokusů |
| `nights` | zpracované noci + statistiky (kolik minut vyřazeno kvůli soukromí, prázdné scéně, tmě) |
| `segments` / `frames` | segment každé noci a ke každému snímku jeho čas a skóre |
| `master` / `master_fragments` | přesný stav výsledného souboru (konec dat, délka, index pro převíjení) |
| `redactions` | ručně vyřazené časové úseky |

Díky tomu se nic nestahuje ani nezpracovává dvakrát. Záznamy mimo `capture_hours` se vůbec nestahují.
Po zpracování noci se stažené soubory smažou (`schedule.keep_downloads`).

### Výběr snímků (adaptivní rychlost)

Každý analyzovaný snímek (`analysis.fps`, výchozí 1 za sekundu) dostane skóre pohybu, ostrosti a jasu. Pokud jsou
dostupné modely, dostane i skóre soukromí, přítomnosti a zajímavosti. Podle stavu scény „virtuální hodiny“ určí,
kolik reálného času připadne na jeden výstupní snímek:

| stav scény | jak se pozná | 1 snímek videa za |
|---|---|---|
| **aktivita** | pohyb v obraze (+ `activity_hold_secs`) | `interval_secs` (20 s) |
| **klid** | někdo je v obraze, ale nehýbe se (spánek) | `calm_interval_secs` (90 s), zajímavé chvíle zrychlí na 20 s |
| **prázdno** | CLIP: v obraze nikdo není (bez CLIP: dlouho žádný pohyb) | `idle_interval_secs` (900 s; `0` = úplně vynechat) |
| **tma** | průměrný jas < `dark_threshold` | nikdy |

V rámci každého časového okna vyhrává **nejlepší snímek**, tedy nejostřejší (bez rozmazání pohybem)
a nejzajímavější, ne náhodný.

---

## Soukromí: vyřazování nahoty

Ochrana je postavená tak, aby se do videa nedostalo nic soukromého ani při nejistotě jednotlivých detektorů:

1. **NudeNet** (detektor YOLOv8, model 640m) hledá v každém analyzovaném snímku odhalená prsa, genitálie,
   hýždě a podobně (`privacy.classes`, práh `privacy.threshold = 0.25`, tedy spíš opatrný). Detektor se pouští
   na každý snímek. Výsledek se znovu použije jen tehdy, když se obraz od poslední kontroly prakticky nezměnil.
2. **CLIP jako druhý názor** pozná *pojem* kojení („a woman breastfeeding a baby“) i ve chvílích, kdy
   NudeNet nic konkrétního nenajde, například v černobílém nočním IR obraze. Aby nevyřazoval běžné mazlení,
   porovnává se s kontrastními situacemi („a woman holding a baby“…) a musí kojení vidět opakovaně
   (`clip_min_hits`).
3. **Časové okno:** detekce v čase *t* vyřadí vše v intervalu **⟨t − 2 min, t + 5 min⟩** (`before_secs`, `after_secs`).
   Kojení je souvislá činnost a detektory mezi snímky „blikají“.
4. **Zpožďovací fronta:** vybraný snímek se zapíše do videa až ve chvíli, kdy je jisté, že se v následujících
   `before_secs` neobjevila detekce. Do enkodéru tedy nikdy nedorazí snímek, který by se zpětně ukázal jako soukromý.
5. **Bez detektoru se nic nezpracuje** (`privacy.required = true`).
6. **Ruční pojistka:** žádný detektor není neomylný. Pokud něco proklouzne:

   ```bash
   tapdream redact --from "2026-10-12 23:10" --to "2026-10-12 23:45"
   ```

   Snímky z daného úseku se odstraní z hotového videa (dotčený segment se překóduje a soubor sestaví znovu).
   Úsek zůstane vyřazený i pro budoucí zpracování.

Ladění na vlastních záběrech: vytáhněte problematický okamžik a podívejte se, co si o něm modely myslí:

```bash
tapdream inspect zaznam.ts --at 1830      # sekunda 1830 videa
tapdream inspect snimek.jpg
```

`tapdream nights` ukazuje, kolik minut bylo každou noc vyřazeno. Highlight každé noci je v `data_dir/highlights/`.

---

## „Zajímavost“ snímků

Prakticky to funguje přes **CLIP**, model, který umí porovnat obrázek s textovým popisem (zero-shot).
Při startu se zakódují textové popisy z konfigurace a každý snímek se s nimi porovná:

- **přítomnost**: „miminko spí v postýlce“, „rodič drží miminko“… proti „prázdná postýlka“, „prázdná ložnice“.
  Pozná se tak kamera zapnutá nad prázdnou postelí, kterou pohybová detekce neodliší od spícího miminka.
- **zajímavost**: „usměvavé miminko“, „miminko je vzhůru a rozhlíží se“, „rodina se mazlí“, „rodič líbá
  miminko“… proti běžnému „miminko spí“, „tmavá ložnice“.

Zajímavost se využívá třikrát:

1. **zpomalí časosběr** v zajímavých chvílích (`scene.interest_weight`),
2. **vybere nejlepší snímek** v každém časovém okně (zajímavý a ostrý),
3. ukládá **fotku noci**, nejzajímavější snímek každé noci jako JPEG do `data_dir/highlights/`.

Popisy jsou v `config.toml` (`scene.interest_prompts` atd.) a můžete si je upravit. Pište je anglicky, tomu CLIP
rozumí nejlépe.

---

## Jeden stále stejný soubor

`timelapse.mp4` je **fragmentované MP4** (stejný formát jako streamovací služby):

```
ftyp | moov (hlavička) | moof+mdat | moof+mdat | … | mfra (index pro převíjení)
```

Připojení noci znamená: odříznout starý index `mfra`, dopsat nové fragmenty, napsat nový index a v hlavičce
přepsat délku videa (pevná velikost, 64bitové hodnoty, vydrží desítky let). **Zapisují se jen nová data.**
Index v SQLite zná přesný konec platných dat. Pokud běh spadne uprostřed zápisu, další běh soubor vrátí
do posledního platného stavu.

- Přehrává se všude, kde se přehrávají MP4 (mpv, VLC, prohlížeče, telefony). Převíjení funguje díky indexu `mfra`.
- `tapdream rebuild` soubor sestaví znovu ze segmentů nocí (rychlé, bez překódování).
- `tapdream export kopie.mp4` vytvoří klasické nefragmentované MP4 („faststart“) pro sdílení a archivaci.
- Formát (rozlišení, fps, kodek) se zamkne při první noci. Kdybyste ho chtěli změnit, `tapdream rebuild --reencode`
  překóduje všechny segmenty.
- Každý keyframe nese parametry kodeku, takže připojování funguje i po aktualizaci ovladače nebo ffmpeg.

---

## Příkazy

| příkaz | co dělá |
|---|---|
| `tapdream init` | vytvoří ukázkový `config.toml` |
| `tapdream models fetch [--no-clip]` | stáhne modely |
| `tapdream doctor [--camera]` | zkontroluje ffmpeg, GPU, ONNX Runtime, modely, font, kameru |
| `tapdream camera info` / `camera list [YYYY-MM-DD]` | test spojení, výpis záznamů na SD kartě |
| `tapdream sync` | stáhne nové záznamy |
| `tapdream process` | zpracuje dokončené noci a připojí je k videu |
| `tapdream run` | `sync` + `process` (pro cron/timer) |
| `tapdream daemon` | `run` každých `daemon_interval_secs` |
| `tapdream status` / `nights` | přehled, statistiky po nocích |
| `tapdream inspect <soubor> [--at s]` | co modely vidí na snímku (ladění prahů) |
| `tapdream redact --from … --to …` | vyřadí časový úsek z videa |
| `tapdream reprocess <YYYY-MM-DD>` | znovu zpracuje noc (vyžaduje `keep_downloads = true`) |
| `tapdream rebuild [--reencode]` | znovu sestaví video ze segmentů |
| `tapdream export <soubor>` | klasická MP4 kopie |

Globální volby: `-c/--config <soubor>` (nebo `TAPDREAM_CONFIG`), `-v` (ladicí výpisy).
Ctrl+C dokončí rozpracovaný krok a skončí. Druhé Ctrl+C ukončí program okamžitě.

---

## Automatický běh (systemd)

V adresáři `deploy/` jsou připravené jednotky:

```bash
sudo useradd --system --home /var/lib/tapdream --create-home tapdream
sudo mkdir -p /etc/tapdream && sudo cp config.toml /etc/tapdream/
echo 'TAPDREAM_CLOUD_PASSWORD=…' | sudo tee /etc/tapdream/env && sudo chmod 600 /etc/tapdream/env
sudo cp deploy/tapdream.service /etc/systemd/system/
sudo systemctl enable --now tapdream
journalctl -u tapdream -f
```

V `/etc/tapdream/config.toml` nastavte `data_dir = "/var/lib/tapdream"`. Alternativně lze místo démona spouštět
`tapdream run` každou hodinu (`deploy/tapdream-run.timer`). Pokud ONNX Runtime instalujete do venv, udělejte to
pod uživatelem `tapdream` (`/var/lib/tapdream/.tapdream-venv`) nebo cestu zadejte v `gpu.onnxruntime_lib`.

---

## Konfigurace – co ladit

Kompletní a komentovaná je v [`config.example.toml`](config.example.toml). Nejdůležitější volby:

| volba | výchozí | význam |
|---|---|---|
| `schedule.capture_hours` | `""` | okno, ze kterého se berou záběry (např. `"19:00-08:00"`) |
| `timelapse.crop` | `""` | ořez zdroje `w:h:x:y`, např. jen na postýlku. Miminko pak vyplní celý obraz. |
| `timelapse.interval_secs` / `calm_interval_secs` / `idle_interval_secs` | 20 / 90 / 900 | tempo videa |
| `timelapse.fps`, `width`, `height`, `codec`, `quality` | 30, 1920×1080, hevc, 24 | formát (zamkne se) |
| `analysis.decode` | `auto` | `keyframes` = dekódovat jen klíčové snímky (nejrychlejší); `auto` to zvolí sám podle záznamu |
| `privacy.threshold` | 0.25 | citlivost NudeNet (nižší = opatrnější) |
| `privacy.before_secs` / `after_secs` | 120 / 300 | časové okno kolem detekce |
| `privacy.clip_threshold` / `clip_min_hits` | 0.6 / 2 | citlivost CLIP na kojení |
| `scene.presence_threshold` | 0.35 | pod touto pravděpodobností je scéna „prázdná“ |
| `overlay.template` | `Noc {night} · {date}` | titulek (`{night} {date} {age} {time} {weekday}`), `language = "en"` pro angličtinu |

Typická noc (10 h, převážně klidný spánek) dá zhruba 400–700 snímků, tedy 15–25 s videa. Rok tak vydá asi
2 hodiny vzpomínek. Celkovou délku nejvíc ovlivní `calm_interval_secs`.

---

## Výkon

- **Stahování** používá rychlý režim kamery „download“ (asi 10× rychlejší než reálný čas). Starší firmware dostane
  automaticky režim „playback“.
- **Dekódování:** NVDEC (`-hwaccel cuda`). Pokud mají záznamy klíčový snímek alespoň každé 2 s, dekódují se jen
  klíčové snímky, což je řádově rychlejší. Dekodér běží v samostatném vlákně souběžně s analýzou.
  Když GPU dekódování selže, automaticky se přejde na CPU.
- **Inference:** ONNX Runtime s CUDA (volitelně TensorRT: `gpu.provider = "tensorrt"`). Na statické scéně se
  NudeNet nepouští znovu, dokud se obraz nezmění. CLIP běží jednou za `scene.every_secs`.
- **Kódování:** NVENC (HEVC, `-preset p6`), jinak libx265.
- **Zápis:** připojení noci zapíše jen pár MB, celý soubor se nikdy nepřepisuje.

Orientační měření bez GPU (4 jádra CPU, NudeNet 320n + CLIP na CPU): 40 minut záznamu se zpracuje za 51 s,
tedy asi 47× rychleji než reálný čas. Většinu času zabírají modely, takže s GPU je to výrazně rychlejší.

---

## Řešení problémů

| problém | řešení |
|---|---|
| `invalid authentication data` | špatné heslo k **cloudovému** účtu TP-Link (ne heslo „Camera Account“). Po opakovaných pokusech kamera dočasně blokuje přihlášení. |
| `camera temporarily blocks logins` | počkejte uvedený počet sekund |
| `media server rejected the cloud password` | totéž heslo slouží i pro stahování (port 8800) |
| `media encryption is turned off` | v aplikaci Tapo zapněte šifrování médií (výchozí stav) |
| `ONNX Runtime not found` | viz [instalace](#3-onnx-runtime-s-podporou-gpu), případně `gpu.onnxruntime_lib` |
| `GPU inference unavailable … using the CPU` | chybí CUDA/cuDNN knihovny. Nainstalujte `onnxruntime-gpu[cuda,cudnn]`, ověřte `nvidia-smi`. |
| noc se nezpracovává | `tapdream status`: čeká se na stažení všech záznamů noci a na konec noci + `grace_minutes` |
| ve videu je něco, co tam nemá být | `tapdream redact …`, pak zvažte nižší `privacy.threshold` |
| hub H200 / bateriové kamery (KLAP) | zatím nepodporováno |

Podrobné výpisy: `tapdream -v …`.

---

## Vývoj a testy

```bash
cargo test                       # jednotkové testy (protokol vs. vektory z pytapo, fMP4, výběr snímků, …)
```

Nástroje v `tools/`:

- `mock_camera.py` simuluje kameru Tapo: klasický i TPAP login, šifrované stahování MPEG-TS, starý firmware
  (`--old-firmware`). Umožňuje vyzkoušet celý řetězec bez kamery.
- `gen_test_vectors.py`, `gen_clip_tokens.py` generují referenční vektory z pytapo a open_clip.
- `check_models.py` je referenční předzpracování NudeNet/CLIP v Pythonu pro porovnání s `tapdream inspect`.

Struktura kódu:

```
src/tapo/       protokol kamery (HTTPS klient, klasický login, TPAP/SPAKE2+, port 8800, MPEG-TS)
src/sync.rs     zjišťování a stahování záznamů
src/vision/     ONNX Runtime, NudeNet, CLIP + tokenizer, zpracování obrazu
src/pipeline/   analýza snímků, výběr snímků, zpracování noci, správa výsledného videa
src/video/      ffmpeg (dekodér/enkodér), fragmentované MP4, titulky
src/db.rs       index (SQLite)
```

---

## Licence a poděkování

Kód je pod licencí MIT (viz `LICENSE`).

- Protokol kamery je přenesen z [pytapo](https://github.com/JurajNyiri/pytapo) (MIT, Juraj Nyíri a přispěvatelé).
  TPAP/SPAKE2+ vychází z [freeKC/tapo-v4-protocol](https://github.com/freeKC/tapo-v4-protocol) (MIT).
- Tokenizer a slovník BPE pochází z [OpenAI CLIP](https://github.com/openai/CLIP) (MIT).
- Modely se stahují zvlášť a mají vlastní licence: [NudeNet](https://github.com/notAI-tech/NudeNet) (AGPL-3.0),
  CLIP ViT-B/32 (MIT, OpenAI; ONNX export z clip-as-service).

Viz také `NOTICE`.
