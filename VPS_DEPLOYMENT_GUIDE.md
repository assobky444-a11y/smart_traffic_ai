# VPS Deployment Guide - Step by Step 🚀

## Project Structure 🗂️
مثال مبسط لهيكلة المشروع (مفيدة في دليل النشر):

```text
vehicle_tracking/
├── app.py                          # التطبيق الرئيسي
├── vehicle_tracking.db             # قاعدة البيانات
├── templates/
│   ├── login.html                  # صفحة تسجيل الدخول
│   ├── register.html               # صفحة التسجيل
│   ├── index.html                  # الصفحة الرئيسية (التتبع)
│   ├── request_analysis.html       # صفحة طلب التحليل
│   ├── admin_dashboard.html        # لوحة تحكم الإدارة
│   ├── analysis.html               # صفحة التحليل (admin only)
│   └── line_drawing.html           # صفحة رسم الخطوط (admin only)
└── analysis_requests/              # مجلد طلبات التحليل
    └── [request_id]/
        ├── tracks.csv
        ├── frame1.jpg
        ├── frame2.jpg
        ├── frame3.jpg
        └── results/
            ├── heatmap.png
            ├── overview.png
            ├── tracks.png
            ├── speed.png
            └── direction.png
```

استبدلت الشجرة السابقة بمخطط أكثر شمولاً يتضمن الملفات والمجلدات الأخرى في الجذر:

```text
vehicle_tracking/
├── app.py
├── app_wrapper.py
├── vehicle_tracking.db
├── config.json
├── user_config.json
├── requirements.txt
├── testApp1.py
├── testApp2.py
├── track_relinking.py
├── translate_templates.py
├── final_cleanup.py
├── IMPLEMENTATION_SUMMARY.py
├── USER_SYSTEM_GUIDE.md
├── modal/                   # ملفات أوزان النموذج (.pt)
│   └── *.pt
├── templates/               # واجهة الويب
│   ├── login.html
│   ├── register.html
│   ├── index.html
│   ├── request_analysis.html
│   ├── admin_dashboard.html
│   ├── analysis.html
│   └── line_drawing.html
├── static/                  # CSS / JS
│   ├── style.css
│   └── script.js
├── analysis_requests/       # مجلد طلبات التحليل (يدوياً أو من الويب)
├── unified_output/          # مخرجات التحليل (tracks.csv, frames/, results/)
├── uploads/                 # ملفات مرفوعة من المستخدم
├── logs/                    # سجلات التشغيل
├── .github/
├── .gitignore
├── venv/ or .venv/          # بيئة التطوير المحلية (يُستثنى عادةً من النسخ)
└── __pycache__/
```

## المتطلبات الأساسية 📋

### 1. VPS Server
- **نظام التشغيل**: Ubuntu 20.04 LTS أو 22.04 LTS (الأفضل)
- **المواصفات الدنيا**:
  - RAM: 4GB على الأقل (يفضل 8GB)
  - CPU: 2 Cores على الأقل
  - Storage: 50GB SSD
  - Port 80 و 443 مفتوح

### 2. Domain Name (اختياري)
- مثال: `vehicles.yourdomain.com`
- يجب توجيه DNS إلى IP الخاص بالـ VPS

---

## الخطوة 1️⃣: الاتصال بالـ VPS

### باستخدام SSH من Windows PowerShell:
```powershell
ssh root@YOUR_VPS_IP
# مثال: ssh root@192.168.1.100
```

### أو باستخدام PuTTY (برنامج Windows):
1. حمل PuTTY من: https://www.putty.org
2. أدخل IP الخاص بالسيرفر
3. Port: 22
4. اضغط Open ثم أدخل username وpassword

---

## الخطوة 2️⃣: تحديث النظام

```bash
# تحديث قوائم الحزم
sudo apt update

# ترقية الحزم المثبتة
sudo apt upgrade -y

# تثبيت الأدوات الأساسية
sudo apt install -y build-essential curl wget git vim nano software-properties-common
```

---

## الخطوة 3️⃣: تثبيت Python 3.10+

```bash
# تحقق من إصدار Python
python3 --version

# إذا كان الإصدار قديم (أقل من 3.9)، ثبت Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# تثبيت pip
sudo apt install -y python3-pip

# ترقية pip
python3.10 -m pip install --upgrade pip
```

---

## الخطوة 4️⃣: تثبيت المكتبات الإضافية المطلوبة

```bash
# مكتبات OpenCV
sudo apt install -y libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0

# مكتبات معالجة الصور والفيديو
sudo apt install -y libgl1-mesa-glx libglib2.0-0 ffmpeg

# مكتبات لـ Shapely
sudo apt install -y libgeos-dev

# تثبيت Nginx (Web Server)
sudo apt install -y nginx

# تثبيت Supervisor (إدارة العمليات)
sudo apt install -y supervisor
```

---

## الخطوة 5️⃣: إنشاء مستخدم للتطبيق

```bash
# إنشاء مستخدم جديد (أكثر أماناً من root)
sudo adduser vehicleapp
# أدخل كلمة مرور قوية

# إضافة المستخدم لمجموعة sudo
sudo usermod -aG sudo vehicleapp

# الانتقال للمستخدم الجديد
su - vehicleapp
```

---

## الخطوة 6️⃣: نقل ملفات المشروع للسيرفر

### الطريقة الأولى: باستخدام Git (الأفضل)

```bash
# إذا كان المشروع على GitHub
cd /home/vehicleapp
git clone https://github.com/YOUR_USERNAME/vehicles_counting.git
cd vehicles_counting
```

### الطريقة الثانية: باستخدام SCP من جهازك المحلي

**من PowerShell على جهازك (ليس السيرفر):**
```powershell
# الانتقال لمجلد المشروع
cd "C:\Users\Mobi lap\Documents\Systems\vehicles_counting"

# رفع الملفات للسيرفر
scp -r * vehicleapp@YOUR_VPS_IP:/home/vehicleapp/vehicles_counting/
```

### الطريقة الثالثة: باستخدام FileZilla (GUI)
1. حمل FileZilla: https://filezilla-project.org
2. Host: `sftp://YOUR_VPS_IP`
3. Username: `vehicleapp`
4. Port: `22`
5. اسحب الملفات من اليسار (جهازك) إلى اليمين (السيرفر)

---

## الخطوة 7️⃣: إعداد Virtual Environment

```bash
cd /home/vehicleapp/vehicles_counting

# إنشاء بيئة افتراضية
python3.10 -m venv venv

# تفعيل البيئة
source venv/bin/activate

# ترقية pip داخل البيئة
pip install --upgrade pip

# تثبيت المكتبات من requirements.txt
pip install -r requirements.txt

# إذا ظهرت أخطاء مع torch، ثبت النسخة CPU فقط:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# تأكيد التثبيت
pip list
```

---

## الخطوة 8️⃣: إنشاء المجلدات المطلوبة

```bash
# التأكد من وجود جميع المجلدات
mkdir -p uploads unified_output logs

# إعطاء الصلاحيات
chmod 755 uploads unified_output logs
```

---

## الخطوة 9️⃣: اختبار تشغيل التطبيق

```bash
# تشغيل التطبيق يدوياً للاختبار
python app.py
```

**يجب أن ترى:**
```
============================================================
Advanced Vehicle Tracking System
============================================================
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:5000
```

**اختبر من المتصفح:**
```
http://YOUR_VPS_IP:5000
```

إذا عمل بنجاح، اضغط `Ctrl+C` لإيقافه وننتقل للخطوة التالية.

---

## الخطوة 🔟: إعداد Gunicorn (Production Server)

```bash
# تثبيت Gunicorn
pip install gunicorn

# اختبار Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 app:app
```

إذا عمل بنجاح، اضغط `Ctrl+C`.

---

## الخطوة 1️⃣1️⃣: إعداد Supervisor لتشغيل تلقائي

```bash
# إنشاء ملف تكوين Supervisor
sudo nano /etc/supervisor/conf.d/vehicleapp.conf
```

**ضع هذا المحتوى:**
```ini
[program:vehicleapp]
command=/home/vehicleapp/vehicles_counting/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 --max-requests 1000 --max-requests-jitter 50 app:app
directory=/home/vehicleapp/vehicles_counting
user=vehicleapp
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/vehicleapp/err.log
stdout_logfile=/var/log/vehicleapp/out.log
environment=PATH="/home/vehicleapp/vehicles_counting/venv/bin"
```

**احفظ الملف:**
- اضغط `Ctrl+O` ثم `Enter`
- اضغط `Ctrl+X` للخروج

```bash
# إنشاء مجلد اللوقات
sudo mkdir -p /var/log/vehicleapp
sudo chown vehicleapp:vehicleapp /var/log/vehicleapp

# تحديث Supervisor
sudo supervisorctl reread
sudo supervisorctl update

# بدء التطبيق
sudo supervisorctl start vehicleapp

# التحقق من الحالة
sudo supervisorctl status vehicleapp
```

---

## الخطوة 1️⃣2️⃣: إعداد Nginx (Reverse Proxy)

```bash
# إنشاء ملف تكوين Nginx
sudo nano /etc/nginx/sites-available/vehicleapp
```

**ضع هذا المحتوى:**
```nginx
server {
    listen 80;
    server_name YOUR_VPS_IP;  # أو domain name إذا كان لديك
    
    client_max_body_size 500M;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts للعمليات الطويلة
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /static {
        alias /home/vehicleapp/vehicles_counting/static;
        expires 30d;
    }
}
```

**احفظ واخرج (Ctrl+O ثم Enter ثم Ctrl+X)**

```bash
# تفعيل الموقع
sudo ln -s /etc/nginx/sites-available/vehicleapp /etc/nginx/sites-enabled/

# حذف الموقع الافتراضي
sudo rm /etc/nginx/sites-enabled/default

# اختبار التكوين
sudo nginx -t

# إعادة تشغيل Nginx
sudo systemctl restart nginx

# تفعيل Nginx عند بدء التشغيل
sudo systemctl enable nginx
```

---

## الخطوة 1️⃣3️⃣: إعداد Firewall

```bash
# تفعيل UFW Firewall
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable

# التحقق من الحالة
sudo ufw status
```

---

## الخطوة 1️⃣4️⃣: (اختياري) إعداد HTTPS مع Let's Encrypt

```bash
# تثبيت Certbot
sudo apt install -y certbot python3-certbot-nginx

# الحصول على شهادة SSL (استبدل YOUR_DOMAIN)
sudo certbot --nginx -d YOUR_DOMAIN.com -d www.YOUR_DOMAIN.com

# تجديد تلقائي
sudo systemctl status certbot.timer
```

---

## الخطوة 1️⃣5️⃣: التحقق من التشغيل

### افتح المتصفح:
```
http://YOUR_VPS_IP
# أو
https://YOUR_DOMAIN.com
```

---

## 🔧 أوامر إدارة مفيدة

### إدارة التطبيق:
```bash
# إيقاف التطبيق
sudo supervisorctl stop vehicleapp

# بدء التطبيق
sudo supervisorctl start vehicleapp

# إعادة التشغيل
sudo supervisorctl restart vehicleapp

# عرض الحالة
sudo supervisorctl status

# عرض اللوقات
sudo tail -f /var/log/vehicleapp/out.log
sudo tail -f /var/log/vehicleapp/err.log
```

### إدارة Nginx:
```bash
# إعادة تشغيل
sudo systemctl restart nginx

# إيقاف
sudo systemctl stop nginx

# بدء
sudo systemctl start nginx

# عرض الحالة
sudo systemctl status nginx
```

### تحديث الكود:
```bash
cd /home/vehicleapp/vehicles_counting

# إذا كنت تستخدم Git
git pull origin main

# تفعيل البيئة الافتراضية
source venv/bin/activate

# تحديث المكتبات إذا لزم الأمر
pip install -r requirements.txt --upgrade

# إعادة تشغيل التطبيق
sudo supervisorctl restart vehicleapp
```

---

## 📊 مراقبة الأداء

### استخدام htop:
```bash
sudo apt install -y htop
htop
```

### استخدام Disk:
```bash
df -h
```

### استخدام Memory:
```bash
free -h
```

### استخدام Logs:
```bash
# لوقات التطبيق
tail -f /var/log/vehicleapp/*.log

# لوقات Nginx
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# لوقات النظام
sudo journalctl -u supervisor -f
```

---

## 🔒 نصائح الأمان

### 1. تغيير SSH Port:
```bash
sudo nano /etc/ssh/sshd_config
# غير Port 22 إلى رقم آخر مثل 2222
sudo systemctl restart sshd
```

### 2. منع تسجيل دخول root:
```bash
sudo nano /etc/ssh/sshd_config
# غير PermitRootLogin yes إلى no
sudo systemctl restart sshd
```

### 3. تثبيت Fail2Ban:
```bash
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 4. عمل Backup دوري:
```bash
# Backup اليدوي
tar -czf backup_$(date +%Y%m%d).tar.gz /home/vehicleapp/vehicles_counting

# Backup للملفات المرفوعة فقط
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz /home/vehicleapp/vehicles_counting/uploads
```

---

## 🐛 حل المشاكل الشائعة

### المشكلة: التطبيق لا يعمل
```bash
# فحص اللوقات
sudo supervisorctl status
sudo tail -50 /var/log/vehicleapp/err.log

# إعادة تشغيل
sudo supervisorctl restart vehicleapp
```

### المشكلة: خطأ في رفع الملفات
```bash
# زيادة حجم الملفات المسموح في Nginx
sudo nano /etc/nginx/nginx.conf
# أضف في http block:
client_max_body_size 1000M;

sudo systemctl restart nginx
```

### المشكلة: نفاد الذاكرة
```bash
# تقليل عدد Workers
sudo nano /etc/supervisor/conf.d/vehicleapp.conf
# غير --workers 4 إلى --workers 2

sudo supervisorctl restart vehicleapp
```

### المشكلة: بطء في المعالجة
```bash
# استخدام CPU version من PyTorch
source /home/vehicleapp/vehicles_counting/venv/bin/activate
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
sudo supervisorctl restart vehicleapp
```

---

## 📞 معلومات مهمة

- **مسار المشروع**: `/home/vehicleapp/vehicles_counting`
- **لوقات التطبيق**: `/var/log/vehicleapp/`
- **لوقات Nginx**: `/var/log/nginx/`
- **Port الداخلي**: 5000
- **Port الخارجي**: 80 (HTTP) / 443 (HTTPS)

---

## ✅ Checklist النهائي

- [ ] VPS جاهز ومتصل
- [ ] Python 3.10+ مثبت
- [ ] جميع المكتبات مثبتة
- [ ] ملفات المشروع منقولة
- [ ] Virtual Environment جاهز
- [ ] التطبيق يعمل يدوياً
- [ ] Gunicorn يعمل
- [ ] Supervisor مُعد ومُشغل
- [ ] Nginx مُعد كـ Reverse Proxy
- [ ] Firewall مُفعل
- [ ] (اختياري) SSL مُعد
- [ ] النسخ الاحتياطي مُفعل

---

## 🎉 مبروك!

الآن تطبيقك يعمل على VPS بشكل احترافي!

للدعم أو الأسئلة، تفضل بالتواصل.
