from curl_cffi import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import random
from datetime import datetime

class RiyasewanaSearchScraper:
    def __init__(self):
        self.base_url = "https://riyasewana.com"
        self.search_url = f"{self.base_url}/search/cars"
        self.vehicles = []
    
    def extract_make_model(self, title):
        """Extract make and model from title"""
        title_lower = title.lower()
        makes = ['toyota', 'nissan', 'honda', 'suzuki', 'mitsubishi', 'mazda', 
                 'daihatsu', 'hyundai', 'kia', 'bmw', 'mercedes', 'audi', 
                 'volkswagen', 'isuzu', 'mahindra', 'tata', 'micro', 'perodua',
                 'ford', 'chevrolet', 'mg', 'rover', 'mini', 'subaru', 'lexus']
        
        make = 'Unknown'
        model = 'Unknown'
        
        for m in makes:
            if m in title_lower:
                make = m.capitalize()
                # Extract model (word after make)
                words = title.split()
                for i, word in enumerate(words):
                    if word.lower() == m and i + 1 < len(words):
                        model = words[i + 1]
                        break
                break
        
        return make, model
    
    def scrape_detail_page(self, url):
        """Visit an individual ad page and extract detailed specs from the table"""
        details = {
            'make': 'Unknown',
            'model': 'Unknown',
            'yom': None,
            'mileage': None,
            'gear': 'Unknown',
            'fuel_type': 'Unknown',
            'options': 'Unknown',
            'engine_cc': None,
            'contact': 'Unknown',
            'location': 'Unknown',
            'price': None,
        }
        
        try:
            response = requests.get(url, impersonate="chrome120")
            if response.status_code != 200:
                print(f"Detail page HTTP {response.status_code}")
                return details
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with class 'spec-table'
            
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    # Process cells in pairs: label, value, label, value...
                    i = 0
                    while i < len(cells) - 1:
                        label = cells[i].get_text(strip=True).lower()
                        value = cells[i + 1].get_text(strip=True)
                        
                        if not value or value == '':
                            i += 2
                            continue
                        
                        if label == 'make':
                            details['make'] = value
                        elif label == 'model':
                            details['model'] = value
                        elif label == 'yom':
                            try:
                                details['yom'] = int(value)
                            except ValueError:
                                details['yom'] = value
                        elif label in ('mileage (km)', 'mileage(km)', 'mileage'):
                            try:
                                details['mileage'] = int(value.replace(',', '').strip())
                            except ValueError:
                                details['mileage'] = value
                        elif label == 'gear':
                            details['gear'] = value
                        elif label in ('fuel type', 'fueltype', 'fuel_type'):
                            details['fuel_type'] = value
                        elif label == 'options':
                            details['options'] = value
                        elif label in ('engine (cc)', 'engine(cc)', 'engine cc', 'engine'):
                            try:
                                details['engine_cc'] = int(value.replace(',', '').strip())
                            except ValueError:
                                details['engine_cc'] = value
                        elif label == 'contact':
                            details['contact'] = value
                        elif label == 'price':
                            price_match = re.search(r'([0-9,]+)', value)
                            if price_match:
                                try:
                                    details['price'] = int(price_match.group(1).replace(',', ''))
                                except ValueError:
                                    pass
                        
                        i += 2
            
            # Extract location from the "Posted by" subtitle
            # The location is the text after the last comma
            for elem in soup.find_all(['h2', 'h3', 'p', 'span', 'small']):
                text = elem.get_text(strip=True)
                if 'posted by' in text.lower():
                    # Extract location after the last comma
                    location_match = re.search(r',\s*([A-Za-z\s]+)\s*$', text)
                    if location_match:
                        location = location_match.group(1).strip()
                        if location and len(location) > 1:
                            details['location'] = location
                    break
            
        except Exception as e:
            print(f"Error scraping detail page: {e}")
        
        return details
    
    def extract_listing_basics(self, listing):
        """Extract basic info (title, URL, price) from a search page listing"""
        try:
            # Extract title and URL
            title_elem = listing.find('a', class_='more')
            if not title_elem and listing.find('h2'):
                title_elem = listing.find('h2').find('a')
            
            title = title_elem.text.strip() if title_elem else 'Unknown'
            
            # Get URL to detail page
            url = 'Unknown'
            if title_elem and title_elem.get('href'):
                url = title_elem.get('href')
                if not url.startswith('http'):
                    url = self.base_url + url
            
            # Extract price from listing text
            full_text = listing.get_text()
            price_match = re.search(r'Rs\.?\s*([0-9,]+)', full_text)
            price = int(price_match.group(1).replace(',', '')) if price_match else None
            
            return {
                'title': title,
                'url': url,
                'price': price,
                'full_text': full_text[:500]
            }
        except Exception:
            return None
    
    def scrape_page(self, page=1):
        """Scrape a single search page, then visit each ad's detail page"""
        url = self.search_url if page == 1 else f"{self.search_url}?page={page}"
        print(f"\n Scraping page {page}: {url}")
        
        try:
            response = requests.get(url, impersonate="chrome120")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all listings
                listings = soup.find_all('li', class_='item')
                if not listings:
                    listings = soup.find_all('div', class_='item')
                
                print(f"   Found {len(listings)} listings")
                
                new_vehicles = 0
                for idx, listing in enumerate(listings):
                    basics = self.extract_listing_basics(listing)
                    if not basics or not basics['price'] or basics['url'] == 'Unknown':
                        continue
                    
                    print(f"   [{idx+1}/{len(listings)}] Fetching details for: {basics['title'][:45]}...")
                    
                    # Visit the individual ad page to get detailed specs
                    detail = self.scrape_detail_page(basics['url'])
                    
                    # Build the complete vehicle record
                    # Prefer detail page data; fall back to listing data
                    vehicle = {
                        'title': basics['title'],
                        'price': detail.get('price') or basics['price'],
                        'make': detail.get('make', 'Unknown'),
                        'model': detail.get('model', 'Unknown'),
                        'yom': detail.get('yom'),
                        'mileage': detail.get('mileage'),
                        'gear': detail.get('gear', 'Unknown'),
                        'fuel_type': detail.get('fuel_type', 'Unknown'),
                        'options': detail.get('options', 'Unknown'),
                        'engine_cc': detail.get('engine_cc'),
                        'details': basics['full_text'],
                        'location': detail.get('location', 'Unknown'),
                        'contact': detail.get('contact', 'Unknown'),
                        'url': basics['url'],
                        'scrape_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # Fall back to title-based make/model if detail page didn't have them
                    if vehicle['make'] == 'Unknown':
                        vehicle['make'], vehicle['model'] = self.extract_make_model(basics['title'])
                    
                    self.vehicles.append(vehicle)
                    new_vehicles += 1
                    
                    print(f" {vehicle['make']} {vehicle['model']} | {vehicle['gear']} | {vehicle['fuel_type']} | {vehicle['location']}")
                    
                    # Polite delay between detail page requests
                    delay = random.uniform(1, 2)
                    time.sleep(delay)
                
                print(f"Added {new_vehicles} vehicles from this page")
                return True
            else:
                print(f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def scrape_pages(self, num_pages=2):
        """Scrape multiple search pages"""
        for page in range(1, num_pages + 1):
            success = self.scrape_page(page)
            if not success:
                break
            if page < num_pages:
                delay = random.uniform(3, 5)
                print(f"   Waiting {delay:.1f} seconds before next page...")
                time.sleep(delay)
        
        return self.vehicles
    
    def save_to_csv(self):
        if not self.vehicles:
            print("No data to save")
            return None
        
        filename = f'riyasewana_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        fieldnames = ['title', 'price', 'make', 'model', 'yom', 'mileage', 
                     'gear', 'fuel_type', 'options', 'engine_cc', 'details',
                     'location', 'contact', 'url', 'scrape_date']
        
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.vehicles)
        
        print(f"\n Saved {len(self.vehicles)} vehicles to {filename}")
        return filename

# Run
if __name__ == "__main__":
    print("="*60)
    print("RIYASEWANA DETAILED SCRAPER")
    print("="*60)
    print("Visits each ad page to extract full vehicle specs")
    print("Fields: make, model, yom, mileage, gear, fuel_type,")
    print("        options, engine_cc, location, contact, price")
    print("="*60)
    
    scraper = RiyasewanaSearchScraper()
    
    pages = int(input("\nSearch pages to scrape: ") or "1")
    
    print(f"\n Scraping {pages} search page(s)...")
    print("  (Each ad page is visited individually for full details)")
    
    vehicles = scraper.scrape_pages(num_pages=pages)
    
    if vehicles:
        filename = scraper.save_to_csv()
        print(f"\n Data saved to: {filename}")
        print(f"\n Total vehicles: {len(vehicles)}")
        print("\n Sample data:")
        for v in vehicles[:3]:
            print(f"  • {v['title']}")
            print(f"    Price: Rs {v['price']:,} | YOM: {v['yom']} | Mileage: {v['mileage']}")
            print(f"    Gear: {v['gear']} | Fuel: {v['fuel_type']} | Location: {v['location']}")
            print(f"    Options: {v['options']}")
            print()
    else:
        print("\n No vehicles scraped")