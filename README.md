# Програмний модуль **"sentiment-analyser-en"** | Вступ

**Модуль `sentiment-analyser-en` – "Програмний модуль виявлення емоційного забарвлення текстів інформаційних повідомлень поданих англійською мовою"**, який написаний мовою програмування `Python`, призначений для виявлення емоційного забарвлення текстів інформаційних повідомлень англійською мовою за допомогою навченої засобами [fastText](https://github.com/facebookresearch/fastText) моделі.

Для побудови моделі сентимент-аналізу, яка застосовуються у програмному модулі [sentiment-analyser-en](https://github.com/wdc-molfar/sentiment-analyser-en), було використано функціонал бібліотеки [fastText](https://github.com/facebookresearch/fastText) мови програмування `Python`. Для навчання класифікатора [fastText](https://github.com/facebookresearch/fastText) спочатку була здійснена попередня обробка й формування навчального збалансованого набору даних у вигляді текстових файлів [en.txt](https://github.com/wdc-molfar/sentiment-datasets/tree/main/train), що містить в кожному рядку навчальне речення разом з міткою, що розпочинаються з префікса `__label__` (`__label__good` чи `__label__bad` для маркування речень позитивної чи негативної тональності відповідно). 

Далі було здійснено процес навчання засобами [fastText](https://github.com/facebookresearch/fastText) (функція `fastText.train_supervised()`). Для навчання моделі були налаштовані наступні гіперпараметри: 
``` python
hyper_params = { 
	"lr": 0.35,         # Learning rate
	"epoch": 100,       # Number of training epochs to train for
	"wordNgrams": 3,    # Number of word n-grams to consider during training
	"dim": 155,         # Size of word vectors
	"ws": 5,            # Size of the context window for CBOW or skip-gram
	"minn": 3,          # Min length of char ngram
	"maxn": 20,         # Max length of char ngram
	"bucket": 2014846,  # Number of buckets
	}
```
Кількість циклів навчання – `100`, кількість слів у n-грамах – `3`, розмірність вектора моделі – `155`, розмір контекстного вікна – `5`, найменша допустима кількість символів в слові – `3`, найбільша – `20`. В результаті для тенувального набору даних [en.txt](https://github.com/wdc-molfar/sentiment-datasets/tree/main/train) була отримана навчена модель для сентимент-аналізу текстів інформаційних повідомлень англійською мовою, яка після процедури квантування була збережена у файлі формату `.ftz` - [en.ftz](https://drive.google.com/u/0/uc?id=12MLctdlgzqw_tMYizZOzzY1Ja8AYK2AH&export=download&confirm=t).

Для одержання стисненої квантованої моделі був використаний метод `quantize` функціоналу [fastText](https://github.com/facebookresearch/fastText) з наступними параметрами квантування:
``` python
model.quantize(
	input=None,
	qout=False,
	cutoff=0,
	retrain=False,
	epoch=None,
	lr=None,
	thread=None,
	verbose=None,
	dsub=2,
	qnorm=False,
	)
```
Більш детальний [опис](https://github.com/wdc-molfar/sentiment-datasets) тренувального набору текстових даних, призначеного для навчання моделі сентимент-аналізу текстів англійською мовою та сам [тренувальний](https://github.com/wdc-molfar/sentiment-datasets/tree/main/train) та [тестовий](https://github.com/wdc-molfar/sentiment-datasets/tree/main/test) набір англійських текстових інформаційних повідомлень позитивної та негативної тональності розміщений у репозиторії [**`sentiment-datasets`**](https://github.com/wdc-molfar/sentiment-datasets).

### Зміст
- [Позначення та найменування програмного модуля](#name)
- [Програмне забезпечення, необхідне для функціонування програмного модуля](#software)
- [Функціональне призначення](#function)
- [Опис логічної структури](#structure)
- [Використовувані технічні засоби](#hardware)
- [Виклик та завантаження](#run)
- [Вхідні дані](#inputdata)
- [Вихідні дані](#outputdata)

<a name="name"></a>
<h2>Позначення та найменування програмного модуля</h2>

Програмний модуль має позначення **`"sentiment-analyser-en"`**.

Повне найменування програмного модуля – **"Програмний модуль виявлення емоційного забарвлення текстів інформаційних повідомлень поданих англійською мовою."**.

<a name="software"></a>
<h2>Програмне забезпечення, необхідне для функціонування програмного модуля</h2>

Для функціонування програмного модуля **`sentiment-analyser-en`**, написаного мовою програмування `Python`, необхідне наступне програмне забезпечення:

- `Docker` [v20.10](https://docs.docker.com/engine/release-notes/#version-2010)
- `Kubernetes` [v1.22.4](https://github.com/kubernetes/kubernetes/releases/tag/v1.22.4)

- `python` [v3.6.0](https://www.python.org/downloads/release/python-360/) or newer

```sh
python --version Python 3.6.0
```

пакети:
- `fastText` [v0.9.2](https://github.com/facebookresearch/fastText)
- `json` [v1.6.1](https://github.com/facebookresearch/fastText)

```sh
pip install fastText==0.9.2
pip install jsons==1.6.1
```

Для функціонування програмного модуля **`sentiment-analyser-en`** також необхідно вивантажити у директорію `models` квантовану модель [en.ftz](https://drive.google.com/u/0/uc?id=12MLctdlgzqw_tMYizZOzzY1Ja8AYK2AH&export=download&confirm=t) — модель для сентимент-аналізу англійських текстів інформаційних повідомлень, що була навчена на тренувальний збалансованій вибірці [en.txt](https://github.com/wdc-molfar/sentiment-datasets/tree/main/train)) із сумарно `30000` рядків позитивних та негативних англійських інформаційних повідомлень розмічених відповідними мітками `__label__pos` чи `__label__neg` в кінці кожного рядка.

Для [тестового набору даних](https://github.com/wdc-molfar/sentiment-datasets/tree/main/test) було отримано оцінку якості навчання англійської моделі – `99,7%`, відповідно, в той час, коли [якість навчання моделей для англійської мови](https://doi.org/10.48550/arXiv.1607.01759) варіюється від `60,2%` й не перевищує `98,6%` для відповідно різних навчальних наборів даних.

<a name="function"></a>
<h2>Функціональне призначення</h2>

Програмний модуль **`sentiment-analyser-en`** призначений для виявлення емоційного забарвлення текстів інформаційних повідомлень англійською мовою за допомогою навченої засобами [fastText](https://github.com/facebookresearch/fastText) моделі.

<a name="structure"></a>
<h2>Опис логічної структури</h2>

Програмний модуль **`sentiment-analyser-en`** складається з частин:
- `main.py` — головного скрипта, що викликає наступні підмодулі:
	- `packagesInstaller.py` — підмодуль, що відповідає за перевірку та завантаження необхідних програмних бібліотек, модулів та підмодулів, і складається з наступних функцій:
		- `setup_packeges()` — функції завантаження та установки необхідних для коректного функціонування програмного модуля **`sentiment-analyser-en`** бібліотек та пакетів
	- `сonfigLoader.py` — підмодуль, що здійснює обробку файлу конфігурацій `config.json` і складається з наступних функцій:
		- `load_default_languages()` — функції обробки поля з тегом `langModels` та присвоєння відповідних навчених моделей для сентимент-аналізу (аналізу та виявлення емоційного забарвлення) текстів інформаційних повідомлень
		- `default_int_value()` — функції обробки поля з тегом `predictLimit` та присвоєння мінімального порогового значення передбачення наймовірнішої мітки емоційного забарвлення
	- `modelsLoader.py` — підмодуль, що призначений для завантаження навчених моделей сентимент-аналізу, які визначені за замовчуванням для кожної із мов у файлі конфігурацій `config.json` або у списку мов за замовуванням (у випадку відсутності файла `config.json` чи виникнення проблем з його зчитуванням та обробкою), і містить наступні функції:
		- `load_models()` — функцію завантаження навчених за допомогою засобів [fastText](https://github.com/facebookresearch/fastText) моделей для сентимент-аналізу текстів інформаційних повідомлень. У випадку, коли список мов та відповідних їм моделей відсутній, невизначений чи виникають проблеми зі зчитуванням та обробкою файла конфігурацій `config.json`, то використовується список моделей за замовчуванням, який містить українську, російську та англійську мови та відповідні назви моделей для сентимент-аналізу
		- `load_fasttext_model()` — функція безпосереднього завантаження навченої моделі [fastText](https://github.com/facebookresearch/fastText) для класифікації настроїв
	- `sentimentAnalyser.py` — підмодуль, що відповідає за сентимент-аналіз та виявлення емоційного забарвлення текстів інформаційних повідомлень, і містить наступну функцію:
		- `predict_emotion()` — функцію безпосереднього передбачення емоціного забарвлення тексту відповідною попередньо завантаженою моделлю [fastText](https://github.com/facebookresearch/fastText) для сентимент-аналізу

У програмному модулі **`sentiment-analyser-en`** вивід результату роботи програми здійснюється у стандартний вихідний потік за допомогою команди `print`. Повідомлення про помилки та інші інформаційні сповіщення про результати роботи програмного модуля виводяться у `output.log`.

Програмний модуль **`sentiment-analyser-en`** за допомогою функції `load_models()` програмного підмодуля `modelsLoader.py` здійснює завантаження навченої моделі сентимент-аналізу, далі очікує та зчитує вхідний потік у форматі `JSON`, звідки відбирає поле `text` з текстом інформаційного повідомлення та поле `lang` з мовою інформаційного повідомлення і в результаті опрацювання функцією `predict_emotion()` підмодуля `sentimentAnalyser.py` здійснює сентимент-аналіз вхідного текстового повідомлення й вивід у вихідний потік розширеного вхідного `JSON` з полями `pem` - передбачення емоціного забарвлення тексту та `em`, що містить одну із міток: `Good` - для позитивного емоційного забарвлення, `Bad` - для негативного емоційного забарвлення та `None` - у випадку, коли ймовірність передбачення нижча за встановлений поріг `predictLimit = 0.9`.

В результаті встановлення для кожного класифікатора порогу імовірності приналежності до певного класу більше `0.7` – мітки отримують близько `88%` повідомлень, для порогового значення `0.8` цей показник становить близько `82%`, а для порогу `0.9` – `72%`.

<a name="hardware"></a>
<h2>Використовувані технічні засоби</h2>

Програмний модуль **`sentiment-analyser-en`** експлуатується на сервері (або у хмарі серверів) під управлінням операційної системи типу `Linux` (64-х разрядна). В основі управління всіх сервісів є система оркестрації `Kubernetes`, де всі контейнери працюють з використанням `Docker`.


<a name="run"></a>
<h2>Виклик та завантаження</h2>

Для серверів, які працюють під керівництвом операційних систем сімейства `Windows OS`, виклик програмного модуля **`sentiment-analyser-en`** здійснюється шляхом запуску скрипта `main.py` з використанням команди `python`. Потрібно відкрити командний рядок – термінал `shell` та написати `python main.py`. Важливо, щоб скрипт знаходився або в директорії, з якої запущено командний рядок, або в каталозі, прописаному у змінній середовища `PATH`. 
Тож завантаження програмного модуля **`sentiment-analyser-en`** забезпечується введенням в командному рядку повного імені завантажувальної програми:

```sh
python main.py
```

Для серверів, які працюють під керівництвом `Unix`-подібних операційних систем (наприклад, `Linux`) на початку скрипта `Python` у першому рядку має бути вказаний повний шлях до інтерпретатора:
``` python
#!/usr/bin/python3
```
або
``` python
#!/usr/bin/env python3
```

В результаті запуску скрипта `main.py` програмного модуля **`sentiment-analyser-en`**  здійснюється зчитування вхідного потік у форматі `JSON`, звідки відбирається поле `text` з текстом інформаційного повідомлення та поле `lang` з мовою інформаційного повідомлення та здійснюється подальша його обробка й аналіз відповідною завантаженою моделлю для розпізнавання емоціного забарвлення текстів.

За замовчуванням, дані, отримані в результаті застосування програмної системи, виводяться в стандартний вихідний потік. Також вивід може перенаправлятися із консолі у файл, який зберігаюється у вигляді `.txt` файла. Для цього використовується оператор `>`.
Повна команда виглядає так:
```sh
python main.py fullpath/.../input.txt > output.txt
```
Тут `output.txt` – це текстовий файл, у який записується результат виконання скрипта.

Операція може використовуватися як в операційній системі `Windows OS`, так і в `Unix`-подібних системах.
Якщо файла, в який повинен вивестися результат, не існує – система створить його автоматично.
При використанні оператора `>` вміст файлу, в який відображаються дані, повністю перезаписується. Якщо наявні дані потрібно зберегти, а нові дописати до існуючих, то використовується оператор `>>`:
```sh
python main.py fullpath/.../input.txt >> output.txt
``` 

<a name="inputdata"></a>
<h2>Вхідні дані</h2>

Формат вхідних даних - `JSON`, що зчитується із вхідного потоку, має наступний вигляд:
```JSON
{
  "service": {
   "scraper": {
    "message": {
     "text": "Good, very good text",
     "html": "MyHtml",
     "publishedAt": "2022-04-14 09:30:18",
     "index": 0,
     "md5": "c54eed430a124b9165d429399bf5f49f",
     "lang": "en"
    },
    "page": {
     "title": "MyTitle",
     "description": "MyDescription",
     "image": "MyImage"
    }
   },
   "scheduler": {
    "task": {
     "params": {
      "type": "telegram",
      "channel": "AK47pfl"
     },
     "state": "processed",
     "processedAt": "2022-04-15 17:38:29"
    }
   }
  }
 }
```

<a name="outputdata"></a>
<h2>Вихідні дані</h2>

Формат вихідних даних - `JSON`, що записується у вихідний потік, має наступний вигляд:

```JSON
{
  "service": {
   "scraper": {
    "message": {
     "text": "Good, very good text",
     "html": "MyHtml",
     "publishedAt": "2022-04-14 09:30:18",
     "index": 0,
     "md5": "c54eed430a124b9165d429399bf5f49f",
     "lang": "en",
     "pem": "(('__label__pos', '__label__neg'), array([1.00001001e+00, 1.00000034e-05]))",
     "em": "Good"
    },
    "page": {
     "title": "MyTitle",
     "description": "MyDescription",
     "image": "MyImage"
    }
   },
   "scheduler": {
    "task": {
     "params": {
      "type": "telegram",
      "channel": "AK47pfl"
     },
     "state": "processed",
     "processedAt": "2022-04-15 17:38:29"
    }
   }
  }
 }
```

## Copyright
Copyright © 2022 [WDC-MOLFAR](https://github.com/wdc-molfar)
