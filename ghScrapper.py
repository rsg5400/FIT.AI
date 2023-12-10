from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle

url = 'https://www.ubereats.com/city/state-college-pa?pl=JTdCJTIyYWRkcmVzcyUyMiUzQSUyMjgzMyUyMFclMjBDb2xsZWdlJTIwQXZlJTIyJTJDJTIycmVmZXJlbmNlJTIyJTNBJTIyaGVyZSUzQWFmJTNBc3RyZWV0c2VjdGlvbiUzQTFlc3JlbWhkNVZSYlFVa05HTWpSaUIlM0FFQUlhQXpnek13JTIyJTJDJTIycmVmZXJlbmNlVHlwZSUyMiUzQSUyMmhlcmVfcGxhY2VzJTIyJTJDJTIybGF0aXR1ZGUlMjIlM0EzOS45NTIxNCUyQyUyMmxvbmdpdHVkZSUyMiUzQS03Ni43NDU4MyU3RA%3D%3D&referrer=https%3A%2F%2Fwww.ubereats.com%2Fcity%2Fstate-college-pa'
driver = webdriver.Chrome()
driver.get(url)

resturants= driver.find_elements(By.XPATH, '//a[@data-testid="store-card"]')
resturant_dict = {}
# Loop through the elements and print their "href" attribute
for anchor in resturants:
    href = anchor.get_attribute("href")
    print(href)
    driver.get(href)
    menu_items=driver.find_elements(By.XPATH, '//span[@data-testid="rich-text"]') 
    arr = []
    ## div_elements = driver.find_elements(By.XPATH, '//div[@class="bo bp ev en b1"]')

    for item in menu_items:
        try:
            text_content = item.text
            arr.append(text_content)
            print(text_content)
        except:
            pass

    resturant_dict[href] = arr
    del arr

    driver.back()

with open('menu_items_dict.pickle', 'wb') as file:
    pickle.dump(resturant_dict, file)
#
# while(True):
#     pass
# search = driver.find_element(By.ID, "569512")
#
# print(search)

# elements = driver.find_element(By.CLASS_NAME, 'searchResult fadeIn u-line--light restaurantCardRedesign--desktop-card-wrapper s-col-xl-3 s-col-lg-4 s-col-sm-6 s-col-xs-12')
#
# # restaurant-name u-text-wrap s-card-title--darkLink
# for place in elements:
#     title = place.find_element(By.XPATH, './/*[@id="ghs-search-results-restaurantId-479017"]').text
#     print(title)

# driver.close()
##//*[@id="ghs-search-results-restaurantId-479126"]
# sekarchResult fadeIn u-line--light
# Add an explicit wait for an element to be present on the page
# WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "example-element")))


