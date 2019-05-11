from scrapy.http import Request
from scrapy.selector import HtmlXPathSelector
from scrapy.spider import BaseSpider
import scrapy
product='oneplus'

class GoodsSpider(scrapy.Spider):
    name = 'goods'
    
    def start_requests(self):
        yield Request(url='https://www.flipkart.com/search?q='+product, callback=self.parse_flipkart)

    def parse_flipkart(self, response):
        path='https://www.flipkart.com'
        items=response.xpath('//div[@data-id]')
        for item in items:
            text=item.xpath('.//div/text()').extract()
            print(text)
            if '₹' in text:
                current_price=text[text.index('₹')-1]
                original_price=text[text.index('₹')+1]
            else:
                current_price=text[2]
                original_price=current_price
            link=path+item.xpath('.//*/@href')[0].extract()
            Rating=item.xpath('.//span[contains(@id,"productRating")]/div/text()').extract()
            print(Rating)
            if response.xpath('//div[contains(@style,"grayscale")]').extract_first()!=[]:
                stock="product out of stock"
            else:
                stock="IN STOCK"
            if Rating==[]:
                Rating='NO Rating available'
            else:
                Rating=Rating[0]
            if Rating in text:
                text.remove(Rating)
            if text[0]=='ON OFFER':
             title=item.xpath('.//*/@alt').extract()
            else:
                title=title=item.xpath('.//div/a/@title').extract()
            break  
            print(stock)
        yield {'Website':'Flipkart','Stock':stock,'Product':title,'Rating':Rating,'Original Price':original_price,'Current Price':current_price,'LINK':link}
        yield Request(url='https://www.amazon.in/s?k='+product, callback=self.parse_amazon)

    def parse_amazon(self, response):
         link=response.xpath('//div/a[@class="a-link-normal a-text-normal"]/@href').extract_first()
         title=response.xpath('//h2/text()').extract_first()
         rating=response.xpath('//i[contains(@class,"a-icon-star")]/span/text()').extract_first()
         original_price=response.xpath('//*[contains(@class,"strike")]/text()').extract_first()
         current_price= response.xpath('//*[contains(@class,"price")]/text()').extract_first()
         stock="IN STOCK"
         yield {'Website':'Amazon.in','Stock':stock,'Product':title,'Rating':rating,'Original Price':original_price,'Current Price':current_price,'LINK':link}
         yield Request(url='https://www.snapdeal.com/search?keyword='+product, callback=self.parse_snap)
    
    def parse_snap(self,response):
        link=response.xpath('//a[@pogid]/@href').extract_first()
        title=response.xpath('//p[@class="product-title"]/text()').extract_first()
        current_price=response.xpath('//span[contains(@class,"product-price")]/text()').extract_first()
        original_price=response.xpath('//span[contains(@class,"product-desc-price")]/text()').extract_first()
        rating='NO rating available'
        stock="IN STOCK"
        yield {'Website':'Snapdeal','Stock':stock,'Product':title,'Rating':rating,'Original Price':original_price,'Current Price':current_price,'LINK':link}
        yield Request(url='https://www.shopclues.com/search?q='+product, callback=self.parse_shop)
        
    def parse_shop(self,response):
        s=product.split(' ')
        j='%20'.join(s)
        link=response.xpath('//div[@class="column col3 search_blocks"]/a/@href').extract_first()
        title=response.xpath('//h2/text()').extract_first()
        current_price=response.xpath('//div[@class="ori_price"]/span/text()').extract_first()
        ref=response.xpath('//div[@class="refurbished_i"]/text()').extract_first()
        if ref=='Refurbished':
            original_price=current_price
        else:
            original_price=response.xpath('//div[@class="old_prices"]/span/text()').extract_first()
        rating='NO rating available'
        stock="IN STOCK"
        yield {'Website':'Shopclues.com','Stock':stock,'Product':title,'Rating':rating,'Original Price':original_price,'Current Price':current_price,'LINK':link} 
        yield Request(url='https://paytmmall.com/shop/search?q='+j+'&from=organic&child_site_id=6', callback=self.parse_paytm)
    def parse_paytm(self,response):
        path='https://paytmmall.com'
        items=response.xpath('//div[@class="_2i1r"]')[0]
        print(items.extract())
        current_price=items.xpath('.//div[@class="_1kMS"]/span/text()').extract_first()
        print(current_price)
        if items.xpath('.//div[@class="dQm2"]/span/text()').extract_first()==[]:
         original_price=current_price
        elif items.xpath('.//div[@class="dQm2"]/span/text()').extract_first()=='-' :
         original_price=items.xpath('.//div[@class="dQm2"]/text()').extract_first()
        else:
         original_price=items.xpath('.//div[@class="dQm2"]/span/text()').extract_first()
        link=path+items.xpath('.//div[@class="_3WhJ"]/a/@href').extract_first()
        title=items.xpath('.//div[@class="_2apC"]/text()').extract_first()
        Rating='NO rating available'
        yield {'Website':'Paytm MALL','Stock':stock,'Product':title,'Rating':rating,'Original Price':original_price,'Current Price':current_price,'LINK':link} 