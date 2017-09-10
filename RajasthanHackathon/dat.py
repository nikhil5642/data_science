import scrap
import requests

page=requests.get("http://0.0.0.0:8000/scrap.html")

print(scrap.seed_price(page)) 
