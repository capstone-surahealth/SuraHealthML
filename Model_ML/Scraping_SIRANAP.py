import pandas as pd
from bs4 import BeautifulSoup
import requests, os, time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def get_rs_detail(kode_rs:str)->dict:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    URL = f"https://yankes.kemkes.go.id/app/siranap/tempat_tidur?kode_rs={kode_rs}&jenis=2&propinsi=33"
    page = requests.get(URL, headers={"User-Agent": "Chrome/88.0.4324.182"})
    soup = BeautifulSoup(page.content, "html.parser")
    
    jenis_ruang_list, last_update_list, kapasitas_tempat_tidur_list, jumlah_kosong_list = [],[],[],[]
    for row in soup.find_all("div", class_="col-md-12 mb-2"):
        jenis_ruang = row.find('div', class_='col-md-6 col-12').get_text(strip=True).split("Update")[0]
        last_update = row.find('small').get_text(strip=True).split("Update ")[1]
        last_update = datetime.strptime(last_update, "%d-%m-%Y %H:%M:%S")
        beds = row.find_all('div', class_='text-center pt-1 pb-1')
        kapasitas_tempat_tidur = beds[0].find('div', style='font-size:20px').get_text(strip=True)
        jumlah_kosong = beds[1].find('div', style='font-size:20px').get_text(strip=True)

        # add to list
        jenis_ruang_list += [jenis_ruang]; kapasitas_tempat_tidur_list += [kapasitas_tempat_tidur];
        jumlah_kosong_list += [jumlah_kosong]; last_update_list += [last_update]

    return {"ts": [ts]*len(jenis_ruang_list),
            "kode_rs": [kode_rs]*len(jenis_ruang_list),
            "jenis_ruang": jenis_ruang_list,
            "kapasitas_tempat_tidur": kapasitas_tempat_tidur_list,
            "jumlah_kosong": jumlah_kosong_list,
            "last_update":last_update_list
            }

if __name__ == "__main__":
    
    for i in range(3372000,3372300):
        print(i, end=": ")
        try:
            
            rs_detail = get_rs_detail(f"{i}")

            if len(rs_detail["kode_rs"]) == 0: 
                print("'-'")
                continue

            data = pd.DataFrame().from_dict(rs_detail, orient="index").T
            
            if f"rs_surakarta.csv" in os.listdir(os.getcwd()):
                existing_df = pd.read_csv(f"rs_surakarta.csv")
                existing_df = pd.concat([existing_df, data])
                existing_df.to_csv(f"rs_surakarta.csv", index=False)
            else:
                data.to_csv(f"rs_surakarta.csv", index=False)
            print("'ada'")

        except Exception as e:
            
            print("'tidak ada'", e)
            continue
        
        time.sleep(.25)