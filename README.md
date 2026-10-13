# tapdream 🌙

**Jedno dlouhé časosběrné video z kamery Tapo, které každou nocí (a každým šlofíkem) roste.**

`tapdream` je CLI nástroj pro Linux napsaný v Rustu. Sám si stahuje záznamy ze SD karty kamery Tapo,
vybere z nich snímky a připojí je na konec **jednoho stále stejného souboru** `timelapse.mp4`.
Cílem je video, ve kterém je vidět, jak miminko roste a kolik nocí jste spolu strávili.

- **Noci i šlofíky.** Noc se bere z nočního okna. Přes den si `tapdream` sám najde šlofíky
  (v záběru někdo je a vydrží tam). Krátké průchody místností a prázdnou postýlku přeskočí.
- **Soukromí na prvním místě.** Snímky s nahotou (typicky kojení) se automaticky vyřazují i s časovou rezervou
  kolem nich. Vše běží lokálně na vašem GPU, žádný obraz neopouští počítač.
- **Inteligentní tempo.** Když se něco děje, video běží pomaleji, klidný spánek se zrychlí a prázdné scény se vynechají.
- **„Zajímavost“ snímků.** Model CLIP vybírá v každém časovém okně nejzajímavější a nejostřejší snímek a ukládá
  „fotku dne“.
- **Výkon.** Dekódování na GPU (NVDEC), dekódování jen klíčových snímků, inference na GPU (ONNX Runtime + CUDA),
  kódování na GPU (NVENC). Připojení nového dne zapíše jen nová data, nic se nepřegenerovává.
- **Robustní index.** SQLite eviduje, co se stáhlo, co se zpracovalo a co je ve výsledném videu. Přerušený běh
  (výpadek proudu) se při dalším spuštění sám dorovná.

```
 SD karta kamery ──► stažení (port 8800, AES) ──► index (SQLite) ──► zpracování dne (noc + šlofíky)
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┘
   ▼
 dekódování (NVDEC) ─► analýza snímků ─────────────────► výběr snímků ─────────► kódování (NVENC)
                       • pohyb, ostrost, tma             • adaptivní rychlost      │
                       • NudeNet (nahota)                • šlofík = někdo v záběru ▼
                       • CLIP (přítomnost, kojení,         aspoň 10 min            segment dne (fMP4)
                         zajímavost)                     • nejlepší snímek v okně  │
                                                         • okno soukromí           ▼
                                                     timelapse.mp4 ◄── připojení na konec (append)
```

> **Stav ověření.** Protokol kamery je přenesen z [pytapo](https://github.com/JurajNyiri/pytapo) (verze 3.4.26)
> a otestován proti referenčním vektorům vygenerovaným přímo z pytapo a proti simulátoru kamery
> (`tools/mock_camera.py`). Proti skutečné kameře zatím vyzkoušen nebyl. Pokud něco nepůjde, pomůže
> výstup `tapdream -v camera info`.

---

## Obsah

1. [Instalace (Arch Linux)](#instalace-arch-linux)
2. [První spuštění](#první-spuštění)
3. [Automatický běh (systemd)](#automatický-běh-systemd)
4. [Jak to funguje: den = noc + šlofíky](#jak-to-funguje-den--noc--šlofíky)
5. [Soukromí: vyřazování nahoty](#soukromí-vyřazování-nahoty)
6. [„Zajímavost“ snímků](#zajímavost-snímků)
7. [Jeden stále stejný soubor](#jeden-stále-stejný-soubor)
8. [Příkazy](#příkazy)
9. [Konfigurace – co ladit](#konfigurace--co-ladit)
10. [Výkon](#výkon)
11. [Řešení problémů](#řešení-problémů)
12. [Jiné distribuce](#jiné-distribuce)
13. [Vývoj a testy](#vývoj-a-testy)
14. [Licence a poděkování](#licence-a-poděkování)

---

## Instalace (Arch Linux)

### 1. Ovladač NVIDIA

Podle [Arch Wiki – NVIDIA](https://wiki.archlinux.org/title/NVIDIA), typicky:

```bash
sudo pacman -S nvidia-open nvidia-utils      # pro starší karty viz wiki
nvidia-smi                                   # ověření
```

`nvidia-utils` obsahuje knihovny pro NVDEC/NVENC. Archový `ffmpeg` je s podporou NVENC/NVDEC sestavený.

### 2. ONNX Runtime (běh AI modelů)

AI modely běží přes ONNX Runtime, který `tapdream` načítá dynamicky a sám ho najde v `/usr/lib`:

```bash
pacman -Ss onnxruntime                # ověřte aktuální názvy balíčků v repozitáři extra
sudo pacman -S onnxruntime-cuda       # GPU (stáhne i cuda + cudnn, několik GB)
# nebo jen CPU:  sudo pacman -S onnxruntime
```

Kdyby balíček v repozitáři chyběl nebo jeho verze CUDA neseděla k ovladači, funguje i instalace z pipu do
virtuálního prostředí. `tapdream` ji najde sám a sám načte i knihovny CUDA/cuDNN z pip balíčků:

```bash
python -m venv ~/.tapdream-venv
~/.tapdream-venv/bin/pip install "onnxruntime-gpu[cuda,cudnn]"     # CUDA 13 (ovladač ≥ 580)
```

Jinou instalaci lze zadat přes `gpu.onnxruntime_lib` v konfiguraci nebo proměnnou `ORT_DYLIB_PATH`.
Bez GPU vše funguje na CPU, jen pomaleji.

### 3. tapdream jako balíček (doporučeno)

V repozitáři je `PKGBUILD`, který sestaví právě naklonovanou verzi (jen commitnuté změny):

```bash
sudo pacman -S --needed base-devel git rust
git clone https://github.com/maceeek/tap-dream.git
cd tap-dream/packaging/arch
makepkg -si
```

Balíček nainstaluje:

| cesta | co |
|---|---|
| `/usr/bin/tapdream` | program |
| `/etc/tapdream/config.toml` | konfigurace (`data_dir = "/var/lib/tapdream"`), při aktualizaci se nepřepíše |
| `/var/lib/tapdream` | data (index, modely, segmenty, výsledné video), vlastník uživatel `tapdream` |
| `tapdream.service`, `tapdream-run.timer` | systemd jednotky |

Systémový uživatel `tapdream` vznikne přes `sysusers.d` a datový adresář přes `tmpfiles.d`.
Aktualizace: `git pull && makepkg -si`.

### 3b. Bez balíčku

```bash
sudo pacman -S --needed base-devel rust ffmpeg openssl curl ttf-dejavu
cargo build --release
sudo install -Dm755 target/release/tapdream /usr/local/bin/tapdream
```

### 4. Pomocný alias

Ruční příkazy je nejlepší pouštět stejně jako službu, tedy jako uživatel `tapdream` se stejnou konfigurací
a heslem. Do `~/.bashrc` (nebo `~/.zshrc`):

```bash
alias td='sudo systemd-run --pty --wait --collect --quiet -p User=tapdream -p WorkingDirectory=/var/lib/tapdream \
  -p EnvironmentFile=-/etc/tapdream/env /usr/bin/tapdream -c /etc/tapdream/config.toml'
```

Bez balíčku stačí `tapdream` s vlastním `config.toml` (`tapdream init`).

### 5. Modely

```bash
td models fetch
```

Stáhne se NudeNet 640m + 320n a CLIP ViT-B/32 (~650 MB) do `data_dir/models`. Stahuje se jen jednou:
NudeNet z GitHubu/PyPI, CLIP (OpenAI) z clip-as-service. To je jediný přístup na internet. Veškerá analýza
pak běží offline.

---

## První spuštění

1. Kamera musí mít **SD kartu** a zapnuté **nahrávání** (ideálně nepřetržité, jde to i při detekci pohybu)
   a musí být v LAN.
2. V konfiguraci (`/etc/tapdream/config.toml`, případně `./config.toml`) vyplňte:
   - `camera.host`: IP adresa kamery (dejte jí v routeru pevnou),
   - `general.timezone`, `overlay.birth_date`,
   - `schedule.night_hours` (např. `"19:00-07:00"`) a `schedule.day_boundary`, viz
     [den = noc + šlofíky](#jak-to-funguje-den--noc--šlofíky).
3. **Heslo k účtu TP-Link/Tapo** (cloudový účet, ke kterému je kamera přidaná) nepatří do konfigurace,
   ale do souboru čitelného jen rootem. Kamera ho vyžaduje pro přístup k záznamům, komunikace ale probíhá
   jen lokálně:

   ```bash
   echo 'TAPDREAM_CLOUD_PASSWORD=…' | sudo tee /etc/tapdream/env >/dev/null
   sudo chmod 600 /etc/tapdream/env
   ```

4. Kontrola prostředí (ffmpeg, GPU, ONNX Runtime, modely, font, kamera):

   ```bash
   td doctor --camera
   td camera list            # záznamy na SD kartě za dnešek
   ```

5. První běh ručně (stáhne nové záznamy a zpracuje hotové dny), pak přehled:

   ```bash
   td run
   td status
   ```

> **Pozor na zámek kamery:** po několika chybných přihlášeních kamera dočasně blokuje přihlašování.
> `tapdream` proto při špatném hesle hned skončí a neopakuje pokusy.

---

## Automatický běh (systemd)

```bash
sudo systemctl enable --now tapdream.service      # démon: každou hodinu sync + zpracování
journalctl -u tapdream -f
```

Alternativa bez démona je hodinový časovač: `sudo systemctl enable --now tapdream-run.timer`.
Bez balíčku zkopírujte jednotky z `deploy/` do `/etc/systemd/system/`, vytvořte uživatele
(`sudo systemd-sysusers packaging/arch/tapdream.sysusers`) a adresáře
(`sudo systemd-tmpfiles --create packaging/arch/tapdream.tmpfiles`).

Pokud ONNX Runtime instalujete přes pip, udělejte to do `/var/lib/tapdream/.tapdream-venv`
(`sudo -u tapdream python -m venv …`), nebo cestu zadejte v `gpu.onnxruntime_lib`.

---

## Jak to funguje: den = noc + šlofíky

### Den

Čas je rozdělený na **dny**: den D trvá od `day_boundary` (výchozí **17:00**) dne D do 17:00 dne D+1.
Obsahuje tedy **noc** z D na D+1 a **šlofíky dne D+1**. Hranici zvolte v čase, kdy miminko obvykle nespí,
aby se nerozdělil šlofík. Den se zpracuje, až skončí (+ `grace_minutes`) a jsou staženy všechny jeho
záznamy. Pak se jako jeden segment připojí na konec časosběru.

### Noc

Záběry z `night_hours` (např. `"19:00-07:00"`, víc oken lze oddělit čárkou) se berou celé. Prázdná scéna
se jen zrychlí na 1 snímek za `idle_interval_secs`. Prázdné `night_hours` znamená, že se celý den bere jako noc.

### Šlofíky

Mimo noční okno (`naps = true`, volitelně jen v `nap_hours`, např. `"09:00-11:30, 12:30-16:00"`):

- úsek se bere, jen když je **někdo v záběru** (CLIP: miminko v postýlce, rodič s miminkem…),
- a když to trvá aspoň **`nap_min_minutes`** (10 min). Snímky se do té doby drží stranou. Kratší návštěva
  (rodiče jen prošli místností, převlékání) se zahodí,
- šlofík končí po **`nap_gap_minutes`** (5 min) bez nikoho v záběru,
- **prázdná místnost přes den se nikdy nedostane do videa** (ani zrychleně),
- spánek, který plynule navazuje na noční okno (usnutí v 18:50, dospávání do 7:20), se počítá do noci.

Do obrazu se vypálí titulek: noc `Noc 102 · 10. 10. 2026 · 3 měsíce 9 dní` (`overlay.template`, počítadlo nocí
od `birth_date`), šlofík `Šlofík · 11. 10. 2026 · 3 měsíce 10 dní` (`overlay.nap_template`).

> Šlofíky se poznají spolehlivě jen se scénovým modelem CLIP (`scene.enabled = true`). Bez něj rozhoduje jen
> pohyb, takže je lepší zadat pevná `nap_hours`, nebo šlofíky vypnout (`naps = false`).

### Index

`data_dir/index.db` (SQLite) eviduje:

| tabulka | obsah |
|---|---|
| `recordings` | každý záznam na kameře: časy, stav (`new` → `downloaded` → `purged`, `skipped`, `failed`), počet pokusů |
| `nights` | zpracované dny + statistiky (snímky ze šlofíků, minuty vyřazené kvůli soukromí, prázdné scéně, tmě, krátké návštěvy) |
| `segments` / `frames` | segment každého dne a ke každému snímku jeho čas a skóre |
| `master` / `master_fragments` | přesný stav výsledného souboru (konec dat, délka, index pro převíjení) |
| `redactions` | ručně vyřazené časové úseky |

Díky tomu se nic nestahuje ani nezpracovává dvakrát. Záznamy, které žádný den nepotřebuje, se vůbec
nestahují. Po zpracování dne se stažené soubory smažou (`schedule.keep_downloads`).

### Výběr snímků (adaptivní rychlost)

Každý analyzovaný snímek (`analysis.fps`, výchozí 1 za sekundu) dostane skóre pohybu, ostrosti a jasu. Pokud jsou
dostupné modely, dostane i skóre soukromí, přítomnosti a zajímavosti. Podle stavu scény „virtuální hodiny“ určí,
kolik reálného času připadne na jeden výstupní snímek:

| stav scény | jak se pozná | 1 snímek videa za |
|---|---|---|
| **aktivita** | pohyb v obraze (+ `activity_hold_secs`) | `interval_secs` (20 s) |
| **klid** | někdo je v obraze, ale nehýbe se (spánek) | `calm_interval_secs` (90 s), zajímavé chvíle zrychlí na 20 s |
| **prázdno v noci** | CLIP: nikdo v obraze (bez CLIP: dlouho žádný pohyb) | `idle_interval_secs` (900 s; `0` = vynechat) |
| **prázdno přes den** | totéž mimo noční okno | nikdy |
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
4. **Okraje:** okno funguje i přes hranice nočního okna, šlofíku a dne. Záběry těsně před a za každým
   použitým úsekem se také stáhnou a zkontrolují. Kojení, které začne v 7:01 při nočním okně do 7:00, tak
   vyřadí i poslední snímky noci.
5. **Zpožďovací fronta:** vybraný snímek se zapíše do videa až ve chvíli, kdy je jisté, že se v následujících
   `before_secs` neobjevila detekce. Do enkodéru tedy nikdy nedorazí snímek, který by se zpětně ukázal jako soukromý.
6. **Bez detektoru se nic nezpracuje** (`privacy.required = true`).
7. **Ruční pojistka:** žádný detektor není neomylný. Pokud něco proklouzne:

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

`tapdream nights` ukazuje, kolik minut bylo každý den vyřazeno. Fotka každého dne je v `data_dir/highlights/`.

---

## „Zajímavost“ snímků

Prakticky to funguje přes **CLIP**, model, který umí porovnat obrázek s textovým popisem (zero-shot).
Při startu se zakódují textové popisy z konfigurace a každý snímek se s nimi porovná:

- **přítomnost**: „miminko spí v postýlce“, „rodič drží miminko“… proti „prázdná postýlka“, „prázdná ložnice“.
  Pozná se tak kamera zapnutá nad prázdnou postelí, kterou pohybová detekce neodliší od spícího miminka.
  Podle přítomnosti se poznají i šlofíky.
- **zajímavost**: „usměvavé miminko“, „miminko je vzhůru a rozhlíží se“, „rodina se mazlí“, „rodič líbá
  miminko“… proti běžnému „miminko spí“, „tmavá ložnice“.

Zajímavost se využívá třikrát:

1. **zpomalí časosběr** v zajímavých chvílích (`scene.interest_weight`),
2. **vybere nejlepší snímek** v každém časovém okně (zajímavý a ostrý),
3. ukládá **fotku dne**, nejzajímavější snímek každého dne jako JPEG do `data_dir/highlights/`.

Popisy jsou v konfiguraci (`scene.interest_prompts` atd.) a můžete si je upravit. Pište je anglicky, tomu CLIP
rozumí nejlépe.

---

## Jeden stále stejný soubor

`timelapse.mp4` je **fragmentované MP4** (stejný formát jako streamovací služby):

```
ftyp | moov (hlavička) | moof+mdat | moof+mdat | … | mfra (index pro převíjení)
```

Připojení dne znamená: odříznout starý index `mfra`, dopsat nové fragmenty, napsat nový index a v hlavičce
přepsat délku videa (pevná velikost, 64bitové hodnoty, vydrží desítky let). **Zapisují se jen nová data.**
Index v SQLite zná přesný konec platných dat. Pokud běh spadne uprostřed zápisu, další běh soubor vrátí
do posledního platného stavu.

- Přehrává se všude, kde se přehrávají MP4 (mpv, VLC, prohlížeče, telefony). Převíjení funguje díky indexu `mfra`.
- `tapdream rebuild` soubor sestaví znovu ze segmentů dnů (rychlé, bez překódování).
- `tapdream export kopie.mp4` vytvoří klasické nefragmentované MP4 („faststart“) pro sdílení a archivaci.
- Formát (rozlišení, fps, kodek) se zamkne při prvním dni. Kdybyste ho chtěli změnit, `tapdream rebuild --reencode`
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
| `tapdream process` | zpracuje dokončené dny a připojí je k videu |
| `tapdream run` | `sync` + `process` (pro časovač) |
| `tapdream daemon` | `run` každých `daemon_interval_secs` |
| `tapdream status` / `nights` | přehled, statistiky po dnech (včetně šlofíků) |
| `tapdream inspect <soubor> [--at s]` | co modely vidí na snímku (ladění prahů) |
| `tapdream redact --from … --to …` | vyřadí časový úsek z videa |
| `tapdream reprocess <YYYY-MM-DD>` | znovu zpracuje den (vyžaduje `keep_downloads = true`) |
| `tapdream rebuild [--reencode]` | znovu sestaví video ze segmentů |
| `tapdream export <soubor>` | klasická MP4 kopie |

Globální volby: `-c/--config <soubor>` (nebo `TAPDREAM_CONFIG`), `-v` (ladicí výpisy, mimo jiné průběh šlofíků).
Ctrl+C dokončí rozpracovaný krok a skončí. Druhé Ctrl+C ukončí program okamžitě.

---

## Konfigurace – co ladit

Kompletní a komentovaná je v [`config.example.toml`](config.example.toml). Nejdůležitější volby:

| volba | výchozí | význam |
|---|---|---|
| `schedule.night_hours` | `""` | noční okno, např. `"19:00-07:00"` (prázdné = celý den jako noc) |
| `schedule.day_boundary` | `17:00` | kde končí jeden den (noc + šlofíky po ní) |
| `schedule.naps` / `nap_hours` | `true` / `""` | šlofíky mimo noční okno, volitelně jen v daných oknech |
| `schedule.nap_min_minutes` / `nap_gap_minutes` | 10 / 5 | minimální délka šlofíku, přerušení, které ho ukončí |
| `timelapse.crop` | `""` | ořez zdroje `w:h:x:y`, např. jen na postýlku. Miminko pak vyplní celý obraz. |
| `timelapse.interval_secs` / `calm_interval_secs` / `idle_interval_secs` | 20 / 90 / 900 | tempo videa |
| `timelapse.fps`, `width`, `height`, `codec`, `quality` | 30, 1920×1080, hevc, 24 | formát (zamkne se) |
| `analysis.decode` | `auto` | `keyframes` = dekódovat jen klíčové snímky (nejrychlejší); `auto` to zvolí sám podle záznamu |
| `privacy.threshold` | 0.25 | citlivost NudeNet (nižší = opatrnější) |
| `privacy.before_secs` / `after_secs` | 120 / 300 | časové okno kolem detekce |
| `privacy.clip_threshold` / `clip_min_hits` | 0.6 / 2 | citlivost CLIP na kojení |
| `scene.presence_threshold` | 0.35 | pod touto pravděpodobností je scéna „prázdná“ |
| `overlay.template` / `nap_template` | `Noc {night} · {date}` / `Šlofík · {date}` | titulky (`{night} {date} {age} {time} {weekday}`), `language = "en"` pro angličtinu |

Typická noc (10 h, převážně klidný spánek) dá zhruba 400–700 snímků, tedy 15–25 s videa, a k tomu pár sekund
ze šlofíků. Rok tak vydá asi 2–3 hodiny vzpomínek. Celkovou délku nejvíc ovlivní `calm_interval_secs`.

---

## Výkon

- **Stahování** používá rychlý režim kamery „download“ (asi 10× rychlejší než reálný čas). Starší firmware dostane
  automaticky režim „playback“. Se šlofíky a nepřetržitým nahráváním se stahuje celý den, počítejte s vyšším
  objemem dat po Wi-Fi. Když kamera nahrává jen při pohybu, je to mnohem méně.
- **Dekódování:** NVDEC (`-hwaccel cuda`). Pokud mají záznamy klíčový snímek alespoň každé 2 s, dekódují se jen
  klíčové snímky, což je řádově rychlejší. Dekodér běží v samostatném vlákně souběžně s analýzou.
  Když GPU dekódování selže, automaticky se přejde na CPU.
- **Inference:** ONNX Runtime s CUDA (volitelně TensorRT: `gpu.provider = "tensorrt"`). Na statické scéně se
  NudeNet nepouští znovu, dokud se obraz nezmění. CLIP běží jednou za `scene.every_secs`.
- **Kódování:** NVENC (HEVC, `-preset p6`), jinak libx265.
- **Zápis:** připojení dne zapíše jen pár MB, celý soubor se nikdy nepřepisuje.

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
| `ONNX Runtime not found` | `sudo pacman -S onnxruntime-cuda` nebo pip venv, případně `gpu.onnxruntime_lib` |
| `GPU inference unavailable … using the CPU` | chybí CUDA/cuDNN nebo nesedí verze k ovladači: `nvidia-smi`, `pacman -Qi cuda cudnn`, případně varianta přes pip |
| `GPU decoding failed … retrying on the CPU` | chybí `nvidia-utils` (libnvcuvid) nebo běží jiná verze ovladače než jádro, pomůže restart po aktualizaci |
| den se nezpracovává | `tapdream status`: čeká se na stažení všech záznamů dne a na konec dne + `grace_minutes` |
| šlofík chybí | `tapdream -v reprocess <den>` vypíše nalezené přítomnosti v záběru a proč byly krátké. Zkuste nižší `nap_min_minutes`, nebo `inspect` (presence). |
| ve videu je něco, co tam nemá být | `tapdream redact …`, pak zvažte nižší `privacy.threshold` |
| hub H200 / bateriové kamery (KLAP) | zatím nepodporováno |

Podrobné výpisy: `tapdream -v …`, u služby `journalctl -u tapdream`.

---

## Jiné distribuce

Funguje na každém Linuxu s ffmpeg (s NVENC/NVDEC pro GPU), OpenSSL a ONNX Runtime. Například na Debianu/Ubuntu:
`sudo apt install ffmpeg curl build-essential pkg-config libssl-dev python3-venv fonts-dejavu-core`, ONNX Runtime
přes pip venv (viz výše), sestavení `cargo build --release` a systemd jednotky z `deploy/`.

---

## Vývoj a testy

```bash
cargo test                       # jednotkové testy (protokol vs. vektory z pytapo, fMP4, výběr snímků, šlofíky, …)
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
src/pipeline/   analýza snímků, výběr snímků (šlofíky, soukromí), zpracování dne, správa výsledného videu
src/video/      ffmpeg (dekodér/enkodér), fragmentované MP4, titulky
src/timeutil.rs dny, noční okna, šlofíky, okraje soukromí, popisky
src/db.rs       index (SQLite)
packaging/arch/ PKGBUILD, sysusers, tmpfiles
deploy/         systemd jednotky
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
