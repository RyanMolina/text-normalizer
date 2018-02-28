import scrapy


class CrawlerItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    body = scrapy.Field()


def parse_article(response):
    article_body_selector = '#article-content > p'
    article_title_selector = '#landing-headline > h1::text'
    item = CrawlerItem()
    item['title'] = response.css(article_title_selector).extract_first().replace('\n', '').strip()
    item['link'] = response.url
    body = ""
    for p in response.css(article_body_selector):
        text = p.css('p::text').extract_first()
        body += text + " "
    item['body'] = body.strip()
    return item


class BanderaInquirerSpider(scrapy.Spider):
    name = "phil_star_spider"
    CATEGORIES = ['balita']
    start_urls = ['http://bandera.inquirer.net/{}'.format(category) for category in CATEGORIES]

    def parse(self, response):
        article_selector = '#lmd-box'
        for item in response.css(article_selector):
            article_link_selector = article_selector + ' > a::attr(href)'
            article_link = item.css(article_link_selector).extract_first()
            yield scrapy.Request(article_link, callback=parse_article)

        next_page_selector = '#landing-read-more > a::attr(href)'
        next_page = response.css(next_page_selector).extract()[-1]
        if next_page:
            yield scrapy.Request(next_page, callback=self.parse)
