import scrapy


class CrawlerItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    body = scrapy.Field()


def parse_article(response):
    ARTICLE_BODY_SELECTOR = '.field-item > p'
    ARTICLE_TITLE_SELECTOR = '#page-title::text'

    item = CrawlerItem()
    item['title'] = response.css(ARTICLE_TITLE_SELECTOR).extract_first().replace('\n', '').strip()
    item['link'] = response.url
    body = ""
    for p in response.css(ARTICLE_BODY_SELECTOR):
        text = p.css('p::text').extract_first()
        body += text + " "

    item['body'] = body.strip()

    return item


class PSNSpider(scrapy.Spider):
    name = "phil_star_spider"
    CATEGORIES = ['dr.-love', 'kutob', 'psn-metro', 'psn-opinyon', 'probinsiya', 'psn-showbiz', 'bansa']
    start_urls = ['http://www.philstar.com/ngayon/{}/archive'.format(category) for category in CATEGORIES]

    def parse(self, response):
        ARTICLE_SELECTOR = '.article-teaser-wrapper'
        for item in response.css(ARTICLE_SELECTOR):
            ARTICLE_LINK_SELECTOR = '.article-title > a::attr(href)'
            ARTICLE_LINK = item.css(ARTICLE_LINK_SELECTOR).extract_first()
            yield scrapy.Request(response.urljoin(ARTICLE_LINK), callback=parse_article)

        NEXT_PAGE_SELECTOR = '.pager-next > a::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)
