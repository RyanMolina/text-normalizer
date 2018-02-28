import scrapy


class CrawlerItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    body = scrapy.Field()


def parse_article(response):
    ARTICLE_BODY_SELECTOR = '.article-content > p'
    ARTICLE_TITLE_SELECTOR = '.news-title::text'
    item = CrawlerItem()
    item['title'] = response.css(ARTICLE_TITLE_SELECTOR).extract_first().replace('\n', '').strip()
    item['link'] = response.url
    body = ""
    for p in response.css(ARTICLE_BODY_SELECTOR):
        text = p.css('p::text').extract_first()
        body += text + " "
    item['body'] = body.strip()
    return item


class ABSCBNSpider(scrapy.Spider):
    name = "abscbn_spider"
    CATEGORIES = ['tagalog-news', 'biyahe', 'good-news', 'showbiz', 'krimen', 'abroad', 'paputok', 'balita', 'sunog',
                  'pera', 'umagang-kay-ganda', 'tv-patrol', 'dzmm', 'bandila', 'bmpm']
    start_urls = ['http://news.abs-cbn.com/list/tag/{}'.format(category) for category in CATEGORIES]

    def parse(self, response):
        ARTICLE_SELECTOR = '.articles > article'
        for item in response.css(ARTICLE_SELECTOR):
            ARTICLE_LINK_SELECTOR = 'a::attr(href)'
            ARTICLE_LINK = item.css(ARTICLE_LINK_SELECTOR).extract_first()
            print(ARTICLE_LINK)
            yield scrapy.Request(response.urljoin(ARTICLE_LINK), callback=parse_article)

        NEXT_PAGE_SELECTOR = 'a[title="Next"]::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()[-1]
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)