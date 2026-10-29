# Nanopore basecalling preprocessing

مقایسه شش روش آماده‌سازی سیگنال نانوپور برای بازخوانی DNA با شبکه عصبی یک‌بعدی و CTC.


## روش‌های مقایسه‌شده

| نام در برنامه | پردازش |
|---|---|
| `raw` | جریان واسنجی‌شده بدون پردازش آزمایشی |
| `robust` | میانه و MAD در سطح خوانش |
| `wavelet` | موجک db4، سه سطح و آستانه‌گذاری نرم |
| `robust_wavelet` | بهنجارسازی و سپس موجک |
| `robust_kalman` | بهنجارسازی و سپس کالمن با بهره پایدار |
| `robust_savgol` | بهنجارسازی و سپس ساویتزکی گولای |

مدل شامل کانال‌های ۶۴، ۱۲۸، ۱۹۲ و ۲۵۶ است. ورودی قطعه‌ای با طول ۲۰۰۰ نمونه و خروجی شامل ۵۰۰ موقعیت زمانی و پنج کلاس است. کدگذاری کلاس‌ها: خالی برابر صفر و A، C، G، T به ترتیب برابر ۱ تا ۴.

تقسیم داده در سطح خوانش با نسبت ۷۰، ۱۵ و ۱۵ درصد و بذر ۴۲ انجام می‌شود. تمام مراحل انتخاب و آموزش مدل‌ها قبل از ارزیابی مجموعه آزمون تمام می‌شوند.

## انتخاب‌های بازسازی

جزئیات زیر به دلیل نبود کد اصلی، انتخاب‌های این پیاده‌سازی هستند:

- تابع فعال‌سازی ReLU.
- شروع فیلتر کالمن در کوواریانس پایدار؛ نسبت پیش‌فرض Q/R برابر ۰٫۲۰ است.
- تمدید متقارن مرزها و برآورد نویز از جزئیات ریزترین سطح در موجک.
- انتخاب هدف هر قطعه بر اساس شروع رویدادها در بازه نیمه‌باز قطعه.
- حذف قطعه‌هایی که کاملاً در محدوده همترازی نیستند یا با طول خروجی CTC سازگار نیستند.
- استفاده از زیرمجموعه‌های قابل تنظیم برای بررسی اولیه پارامترها.

در نتیجه، تعداد خوانش‌ها و قطعه‌های پذیرفته‌شده و نتیجه اجرای جدید ممکن است با پایان‌نامه متفاوت باشد. اثرانگشت ذخیره‌شده مربوط به فهرست شناسه خوانش‌هاست، نه هش کامل محتوای سیگنال‌ها. بذر ثابت نیز تضمین برابری عددی بین سخت‌افزارها و نسخه‌های متفاوت نیست.

## ورودی‌ها

برای اجرای واقعی به FAST5، فایل‌های FASTQ متناظر و توالی مرجع FASTA نیاز است. مرحله Python از FAST5 تک‌خوانشی استفاده می‌کند که Tombo با موفقیت روی آن اجرا شده باشد. برنامه DNA را پشتیبانی می‌کند؛ RNA و POD5 ورودی این نسخه نیستند.

مسیر پیش‌فرض اطلاعات Tombo:

```text
Analyses/RawGenomeCorrected_000/BaseCalled_template/Events
```

طبق قالب Tombo، شروع رویدادها با `read_start_rel_to_raw` به مختصات سیگنال کامل تبدیل می‌شود. رویدادها در ترتیب خوانش استفاده می‌شوند و رشته منفی دوباره معکوس و مکمل نمی‌شود.

## نصب

محیط مدل برای Linux و Python 3.10 یا جدیدتر در نظر گرفته شده است. برای ارزیابی واقعی، نصب موفق `edlib` ضروری است؛ در برخی محیط‌ها نصب آن به ابزار ساخت C++ نیاز دارد.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

نسخه PyTorch متناسب با GPU و درایور را از [راهنمای رسمی](https://pytorch.org/get-started/locally/) نصب کنید. نمونه نسخه CUDA 12.6:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

برای ثبت نسخه‌های واقعی محیط اجرا:

```bash
python -m pip freeze > environment-lock.txt
```

### محیط جدا برای Tombo

Tombo از وابستگی‌های قدیمی استفاده می‌کند؛ محیط آن را از محیط مدل جدا نگه دارید. دستورهای زیر نقطه شروع نصب هستند و خود نصب Tombo در این محیط محلی آزموده نشده است:

```bash
conda create -n nanopore_tombo python=3.7 -y
conda activate nanopore_tombo
conda install -c conda-forge -c bioconda ont-tombo -y
python -m pip install ont-fast5-api
```

## آماده‌سازی فایل‌های خام

مسیرهای زیر نمونه‌اند و باید با مسیر داده واقعی جایگزین شوند. خروجی تبدیل باید پوشه‌ای جدا از ورودی باشد.

```bash
multi_to_single_fast5 \
  --input_path /data/multi_fast5 \
  --save_path /data/single_fast5 \
  --recursive --threads 4

# For uncompressed FASTQ files:
cat /data/fastq/*.fastq > /data/reads.fastq

tombo preprocess annotate_raw_with_fastqs \
  --fast5-basedir /data/single_fast5 \
  --fastq-filenames /data/reads.fastq

tombo resquiggle /data/single_fast5 /data/lambda_reference.fasta \
  --processes 4 --num-most-common-errors 5
```

دو دستور Tombo اطلاعات را داخل FAST5های تک‌خوانشی خروجی می‌نویسند. فایل‌های چندخوانشی ورودی را جدا نگه دارید. اگر FASTQها فشرده باشند، پیش از ادغام آن‌ها را باز کنید.

## آماده‌سازی مجموعه‌داده

در محیط مدل و از ریشه مخزن:

```bash
python nanopore_project.py prepare \
  --fast5 /data/single_fast5 \
  --work /data/nanopore_dataset
```

پوشه `--work` باید جدید باشد. گزارش رد و پذیرش در `qc.csv` و خلاصه آن در `qc_summary.json` ذخیره می‌شود. خوانش‌های آماده‌شده در `reads/` و تقسیم ثابت داده در `manifest.json` قرار می‌گیرند.

## آموزش و ارزیابی

با تنظیم‌های گزارش‌شده برای فیلترها:

```bash
python nanopore_project.py run \
  --work /data/nanopore_dataset \
  --run-name experiment_01 \
  --epochs 20 --batch 128 --workers 2 \
  --bootstrap-repeats 5000
```

برای انتخاب دوباره پارامترها فقط بر اساس آموزش و اعتبارسنجی:

```bash
python nanopore_project.py run \
  --work /data/nanopore_dataset \
  --run-name experiment_with_tuning \
  --tune --pilot-train-reads 1000 --pilot-val-reads 200 \
  --pilot-epochs 1 --epochs 20 --batch 128 --workers 2
```

انتخاب دوباره پارامترها ممکن است به نتیجه‌ای متفاوت از تنظیم‌های پایان‌نامه برسد. برای هر اجرا نام جدیدی بدهید؛ برنامه پوشه مدل‌های قبلی را بازنویسی نمی‌کند. فضای ذخیره‌سازی باید گنجایش داده خام و نسخه واسنجی‌شده آن را داشته باشد. پردازش‌ها در سطح خوانش محاسبه می‌شوند و نسخه کامل جداگانه‌ای برای هر شش روش روی دیسک ساخته نمی‌شود.

## ارزیابی و خروجی‌ها

خروجی قطعه‌ها ابتدا با CTC حریصانه رمزگشایی و سپس به ترتیب در سطح خوانش به هم متصل می‌شوند. مرجع ارزیابی از اتصال هدف قطعه‌های پذیرفته‌شده همان خوانش ساخته می‌شود؛ بنابراین ارزیابی، بخش‌های کنارگذاشته‌شده سیگنال را پوشش نمی‌دهد.

همترازی سراسری با edlib انجام می‌شود. اگر M تعداد انطباق، X جایگزینی، I اضافه و D حذف باشد:

```text
Identity = 100 * M / (M + X + I + D)
Error = 100 - Identity
```

معیار کلی از مجموع شمارش‌ها محاسبه می‌شود، نه میانگین ساده درصدهای خوانش‌ها. در بوت‌استرپ، همان شناسه‌های بازنمونه‌گیری‌شده برای همه روش‌ها استفاده می‌شوند. این بازه‌ها فقط عدم قطعیت خوانش‌های آزمون را می‌سنجند و تغییرپذیری آموزش مجدد را پوشش نمی‌دهند.

| فایل | محتوا |
|---|---|
| `best.pt` | بهترین وضعیت مدل بر اساس اعتبارسنجی |
| `history.json` | زیان آموزش و اعتبارسنجی هر دوره |
| `predictions.fasta` | توالی‌های پیش‌بینی‌شده |
| `read_counts.npz` | شناسه خوانش و شمارش M، X، I، D |
| `metrics.json` | معیارهای نهایی هر روش |
| `summary.json` | مقایسه شش روش |
| `bootstrap.json` | بازه‌های اطمینان و اختلاف با بهنجارسازی |
| `settings.json` | تنظیم‌ها و اطلاعات اصلی محیط |

اجرای مجدد فقط مرحله آماری:

```bash
python nanopore_project.py bootstrap \
  --work /data/nanopore_dataset \
  --run-name experiment_01 --bootstrap-repeats 5000
```

## آزمون‌ها و وضعیت اعتبارسنجی

```bash
python -m unittest discover -s tests -v
```

آزمون‌ها از داده ساختگی استفاده می‌کنند و نیازی به فایل پایان‌نامه یا نمونه زیستی ندارند. هفت آزمون محلی موفق شدند و آزمون همترازی به علت نصب نبودن edlib صریحاً اجرا نشد (skip)؛ برای آزمون کامل، edlib باید نصب باشد. گردش‌کار GitHub Actions همه آزمون‌ها را در Linux با edlib اجرا می‌کند؛ موفقیت آن باید از نتیجه واقعی اجرای Actions بررسی شود.

در بررسی محلی، آماده‌سازی ۲۰ FAST5 ساختگی، آموزش یک‌دوره‌ای شش مدل، رمزگشایی، بازسازی خوانش و بوت‌استرپ بررسی شدند. خود Tombo و edlib در آن محیط قابل اجرا نبودند؛ برای بررسی مسیر ارزیابی، از همتراز مستقل آزمایشی استفاده شد. اجرای کامل روی داده واقعی و بازتولید نتایج پایان‌نامه تأیید نشده است.

## منابع

- پایان‌نامه «طراحی و توسعه سیستم ترکیبی پردازش سیگنال و شبکه‌های عصبی یک‌بعدی جهت ترجمه سیگنال‌های خام توالی‌یاب نانوپور»، بهروز زمانی‌پور قهرود، ۱۴۰۵.
- [مطالعه مرجع مجموعه‌داده و مقایسه بازخوان‌ها](https://doi.org/10.1186/s13059-023-02903-2)
- [قالب و همترازی Tombo](https://nanoporetech.github.io/tombo/resquiggle.html)
- [تبدیل فایل‌های FAST5](https://github.com/nanoporetech/ont_fast5_api)
- [PyTorch CTCLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
- [edlib](https://github.com/Martinsos/edlib)

## English overview

Reconstructed implementation of a thesis comparing six read-wise signal preprocessing strategies for R9.4.1 nanopore DNA basecalling. It provides single-read Tombo FAST5 ingestion, calibrated current, read-level splits, 2,000-sample chunks, a fixed 1D CNN with CTC, validation-based selection, read-level global alignment, and paired bootstrap intervals. The historical source, dataset, checkpoints and original run outputs are not included. Reconstruction choices and validation limits are documented above.
