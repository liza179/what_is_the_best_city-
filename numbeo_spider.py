import scrapy
from scrapy.crawler import CrawlerProcess

class NumbeoSpider(scrapy.Spider):
    name = 'numbeo_spider'
    start_urls = ['https://www.numbeo.com/cost-of-living/rankings_by_country.jsp']

    def parse(self, response):
        for row in response.xpath('//*[@id="t2"]/tbody/tr'):
            yield {
                'Country': row.xpath('td[2]//text()').get(),
                'Cost of Living Index': row.xpath('td[3]//text()').get(),
                'Rent Index': row.xpath('td[4]//text()').get(),
                'Cost of Living Plus Rent Index': row.xpath('td[5]//text()').get(),
                'Groceries Index': row.xpath('td[6]//text()').get(),
                'Restaurant Price Index': row.xpath('td[7]//text()').get(),
                'Local Purchasing Power Index': row.xpath('td[8]//text()').get()
            }

# Запуск процесса
process = CrawlerProcess(settings={
    'FEED_FORMAT': 'csv',         # Сохраняем в формате CSV
    'FEED_URI': 'numbeo_data.csv' # Название выходного файла
})

process.crawl(NumbeoSpider)
process.start()  # Скрипт будет блокирован здесь до завершения работы паука
