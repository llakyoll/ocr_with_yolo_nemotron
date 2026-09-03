# Git LFS ile Video Aktarımı Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kaynak videoların GitHub üzerinden Git LFS ile bilgisayarlar arasında aktarılmasını sağlamak.

**Architecture:** Video uzantıları Git LFS işaretçileri olarak Git'te tutulur; büyük içerik GitHub LFS depolamasına gönderilir. Kaynak videolar `data/input/` altında izlenir, çıktı ve gizli dosyalar Git dışında kalır.

**Tech Stack:** Git, Git LFS 3.0.2, GitHub.

## Global Constraints

- LFS yalnızca `*.mp4`, `*.mov`, `*.avi` ve `*.mkv` dosyalarını izler.
- Kaynak videolar `data/input/` altında bulunur.
- `runs/`, `models/`, `.env` ve OCR ara çıktıları Git'e eklenmez.
- Kurulum gerçek bir video dosyası eklemez; ilk video kullanıcı tarafından seçilir.

---

### Task 1: Git LFS video izleme kuralları

**Files:**
- Create: `.gitattributes`
- Create: `.gitignore`

**Interfaces:**
- Consumes: Yerel `git-lfs` kurulumu.
- Produces: LFS video uzantısı kuralları ve çalışma dosyası dışlamaları.

- [ ] **Step 1: LFS kuralını oluşturun**

```bash
git lfs track "*.mp4" "*.mov" "*.avi" "*.mkv"
```

- [ ] **Step 2: Kuralı doğrulayın**

```bash
git check-attr filter diff merge -- example.mp4
```

Expected: Üç öznitelik de `lfs` değerini gösterir.

- [ ] **Step 3: Çalışma çıktıları dışlamalarını ekleyin**

```gitignore
runs/
models/
.env
.venv/
__pycache__/
*.py[cod]
data/ground_truth/*.private.*
```

- [ ] **Step 4: Kuralları commit edin**

```bash
git add .gitattributes .gitignore
git commit -m "build: configure Git LFS video tracking"
```

### Task 2: Kullanım belgeleri ve proje planı

**Files:**
- Create: `README.md`
- Modify: `PLAN.md`

**Interfaces:**
- Consumes: Task 1 LFS uzantı kuralları.
- Produces: İki bilgisayardaki kurulum, yükleme ve indirme komutları.

- [ ] **Step 1: İlk bilgisayar akışını belgeleyin**

```bash
git lfs install
mkdir -p data/input
cp /path/to/video.mp4 data/input/
git add data/input/video.mp4
git commit -m "data: add input video"
git push origin main
```

- [ ] **Step 2: İkinci bilgisayar akışını belgeleyin**

```bash
git lfs install
git clone https://github.com/llakyoll/ocr_with_yolo_nemotron.git
cd ocr_with_yolo_nemotron
git lfs pull
```

- [ ] **Step 3: PLAN.md'ye kota notunu ekleyin**

```markdown
Kaynak videolar Git LFS ile sürümlenir. GitHub LFS depolama ve bant genişliği
kotaları izlenir; büyük ham veri setleri harici depolamaya taşınır.
```

- [ ] **Step 4: Belgeleri commit edin**

```bash
git add README.md PLAN.md
git commit -m "docs: document Git LFS video workflow"
```

### Task 3: Kurulumu doğrulama ve GitHub'a gönderme

**Files:**
- Modify: Git geçmişi (Task 1 ve Task 2 commit'leri gönderilir)

**Interfaces:**
- Consumes: Yapılandırma ve belgeleme commit'leri, `origin` remote'u.
- Produces: GitHub'da LFS kuralları ve kullanım belgeleri.

- [ ] **Step 1: LFS ortamını doğrulayın**

```bash
git lfs env
git lfs track
git check-ignore -v runs/annotated.mp4 models/license_plate_yolo.pt .env
```

Expected: Dört video uzantısı LFS takip listesinde, çalışma örnekleri `.gitignore` tarafından dışlanır.

- [ ] **Step 2: Commit edilmiş kuralları gözden geçirin**

```bash
git status --short
git log --oneline -3
git show --check --stat HEAD
```

Expected: Çalışma dizini temizdir ve ilgili commit'ler geçmişte görünür.

- [ ] **Step 3: GitHub'a gönderin**

```bash
git push origin main
```
