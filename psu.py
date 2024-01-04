import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By  # Add this import statement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


driver = webdriver.Chrome()
link= 'https://menu.hfs.psu.edu/shortmenu.aspx?sName=Penn+State+Housing+and+Food+Services&locationNum=11&locationName=East+Food+District&naFlag=1&WeeksMenus=This+Week%27s+Menus&myaction=read&dtdate=1%2f8%2f2024'
driver.get(link)
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

anchor_tags = soup.find_all('a')

# Extract and print href attributes
for anchor_tag in anchor_tags:
    href = anchor_tag.get('href')

url = 'https://menu.hfs.psu.edu/longmenu.aspx?sName=Penn+State+Housing+and+Food+Services&locationNum=11&locationName=East+Food+District&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=1%2f8%2f2024&mealName=Breakfast'

try:
    # Open the specified URL
    driver.get(url)
    # Find all checkbox elements on the page
    # checkboxes = driver.find_element(By.XPATH, '//*[@id="menuDisplay"]/tbody/tr[3]/td[1]/table/tbody/tr/td[1]/input').click()
    # checkboxes = driver.find_elements(By.XPATH, '//*[@id="menuDisplay"]/tbody/tr[3]/td[1]/table/tbody/tr/td[1]/input')
    # checkboxes = driver.find_elements(By.TAG_NAME, "input")
    checkboxes = driver.find_elements(By.CSS_SELECTOR, 'input[type="checkbox"]')


    time.sleep(2)
    # Check each checkbox
    for checkbox in checkboxes:
        print('here')
    
        try:
        # Do something with each checkbox (optional)
            # print("Checkbox state:", checkbox.is_selected())
    
        # Click the checkbox
            checkbox.click()
        except Exception as e:
            print(e)


    showReport = driver.find_element(By.CSS_SELECTOR, 'input[type="button"]').click()

    # table = driver.find_elements(By.TAG_NAME, 'table')[1].text
    # print(table)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    table = soup.find_all('table')

    print(table[1])
            


finally:
    # Close the browser window
    driver.quit()
