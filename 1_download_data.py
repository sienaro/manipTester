"""
Step 1: Download MentalManip dataset and Miller Center speeches.
Run: python 1_download_data.py
"""
import urllib.request, subprocess

print("Downloading MentalManip...")
for fname in ["mentalmanip_maj.csv", "mentalmanip_con.csv"]:
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/audreycs/MentalManip/main/mentalmanip_dataset/{fname}",
        fname
    )
print("Done. mentalmanip_maj.csv and mentalmanip_con.csv saved.")

print("Downloading Miller Center speeches (~200MB)...")
urllib.request.urlretrieve("https://data.millercenter.org/miller_center_speeches.tgz",
                           "miller_center_speeches.tgz")
subprocess.run(["tar", "-xzf", "miller_center_speeches.tgz"], check=True)
print("Done. speeches/ folder extracted.")

print("\nNOTE: speech_labels_final.csv (hand-labeled data) must be copied from the repo.")
