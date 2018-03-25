import scrapy


class CrawlerItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    body = scrapy.Field()


def parse_article(response):
    ARTICLE_BODY_SELECTOR = '.td-post-content > p'
    ARTICLE_TITLE_SELECTOR = '.entry-title::text'
    item = CrawlerItem()
    item['title'] = response.css(ARTICLE_TITLE_SELECTOR).extract_first().replace('\n', '').strip()
    item['link'] = response.url
    body = ""
    for p in response.css(ARTICLE_BODY_SELECTOR):
        text = p.css('p::text').extract_first()
        body += text + " "
    item['body'] = body.strip()
    return item


class AbanteToniteSpider(scrapy.Spider):
    name = "phil_star_spider"
    CATEGORIES = ['balitang-promdi',
                  'local-news',
                  'sports']
    start_urls = ['http://www.abante-tonite.com/{}'.format(category) for category in CATEGORIES]

    def parse(self, response):
        ARTICLE_SELECTOR = '.td_module_wrap'
        for item in response.css(ARTICLE_SELECTOR):
            ARTICLE_LINK_SELECTOR = '.td-module-title > a::attr(href)'
            ARTICLE_LINK = item.css(ARTICLE_LINK_SELECTOR).extract_first()
            yield scrapy.Request(ARTICLE_LINK, callback=parse_article)

        NEXT_PAGE_SELECTOR = '.page-nav > a::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()[-1]
        if next_page:
            yield scrapy.Request(next_page, callback=self.parse)
