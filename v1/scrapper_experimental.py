from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
from urllib.parse import urljoin
import requests
import time


base_url = "https://docs.manim.community/en/latest/"
starting_page = "reference_index/animations.html"

def crawl(url: str, depth: int = 10):
  current_url = url
  pages_crawled = 0

  while pages_crawled < depth and current_url:
    try:
      response = requests.get(current_url)
      if response.status_code != 200:
        print(f"Failed to retrieve {current_url}")
        break
      soup = BeautifulSoup(response.content, 'html.parser')
      yield scrape(soup)
      pages_crawled += 1
      next_link = soup.find('a', class_='next-page')
      if next_link:
        next_href = next_link.get('href')
        if next_href:
          current_url = urljoin(current_url, next_href)
          #TODO: check if url is valid

        else: 
          break
      else: 
        break

      time.sleep(.5)

    except requests.exceptions.RequestException as e:
      print(f"Error fetching {current_url}: {e}")
    # except Exception as e:
    #   print(f"Unexpected error while processing {current_url}: {e}")

def scrape(soup):
  try:
    section = soup.find("section")
    if section:
      markdown = convert_to_markdown(section)
      markdown = '\n'.join(line for line in markdown.splitlines() if line.strip() != '')
      return markdown
    else:
      print(f"No section tag found in soup")
      return soup.prettify()
  except Exception as e:
    print(f"Failed to convert section to markdown for soup: {str(e)}")
    return soup.prettify()


def main():
  url = base_url + starting_page
  print("Printing scrapped data!")
  for markdown in crawl(url, depth=3):
    print("\n" + "="*80 + "\n")
    print(markdown)
    print("\n" + "="*80 + "\n")

main()
