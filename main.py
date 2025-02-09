# # import nltk

# # # تحديد مسار التنزيل داخل البيئة الافتراضية
# # nltk.data.path.append("E:/Task7/mage-env/nltk_data")

# # # تنزيل punkt
# # nltk.download("punkt", download_dir="E:/Task7/mage-env/nltk_data")
import nltk
from nltk.tokenize import sent_tokenize

# تحديد مسار nltk_data يدوياً
nltk.download("punkt", download_dir="E:/Task7/mage-env/nltk_data")

text = "Hello! This is a test. How are you?"
print(sent_tokenize(text))
