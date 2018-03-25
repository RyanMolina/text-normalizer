import json
import re
import scrapy


class CrawlerItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    body = scrapy.Field()


class GMASpider(scrapy.Spider):
    name = "gma_spider"
    start_urls = ['http://www.gmanetwork.com/news/archives/get_archives/news/{}/1/ulatfilipino/'
                  .format(year) for year in range(2007, 2018)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var_pattern = re.compile(r'var initialData = (.*)')

    def parse(self, response):
        data = json.loads(response.body_as_unicode())
        for item in data:
            article_link = item.get('url')
            yield scrapy.Request(article_link, callback=self.parse_article)

        tokens = response.url.split('/')
        tokens[-3] = str(int(tokens[-3]) + 1)
        next_page = '/'.join(tokens)

        yield scrapy.Request(next_page, callback=self.parse)

    def parse_article(self, response):
        script = scrapy.selector.Selector(text=response.body).css('script').extract()[-4]
        m = self.var_pattern.search(script)
        article_data = json.loads(m.group(0).replace('var initialData = ', '')[:-1])

        item = CrawlerItem()
        item['title'] = article_data['story']['title']
        item['link'] = response.url

        main = article_data['story']['main']
        paragraphs = scrapy.selector.Selector(text=main).css('::text').extract()

        body = ""
        for p in paragraphs:
            body += p + " "
        item['body'] = body.rstrip(' ')
        return item
