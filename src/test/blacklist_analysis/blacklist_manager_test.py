from managers.blacklist_manager import BlacklistManager

TEST_CASE = "我於【Tinder】得知投資訊息，在【Tinder】以「單純交友為前提」認識歹徒，歹徒要我加入對方的LINE ID，慫恿至【假投資網站投資（網站名稱:ousdinh及網址:https://www.ousdinh.com）】，誆稱保證獲利、穩賺不賠，我遂依指示從轉帳10000元至，並至該網站申請帳號錢包〈Bitget Wallet〉，並依照歹徒指示【轉帳、購買虛擬貨幣】，惟因【對方後續要我轉帳愈來愈多金錢】，驚覺受騙從錢包〈Bitget Wallet〉轉帳10000至我的戶頭，計損失新臺幣【0元】。我因有求職需求，在網路【IG】發現【賣場小編】工作機會，便加入對方LINE 【LINE ID:@522pqueg】，對方稱說要【工作認證】，要求提供【身分證及帳戶等照片】，並指示以【超商店到店】寄送【金融卡】，我依指示寄出後，結果後續有人轉2筆金額近來，我便驚覺怪異，且銀行通知帳戶遭警示，方知被騙，故至所報案。我稱於【114年06月11日11時08分】收到【電子郵件nick.roelstraete@telent.be】假冒【PQD.AR-服務FETC.Service(遠通電收)】名義之釣魚簡訊，簡訊內容稱【尚未繳納停車費新台幣120元】，並傳送連結惡意連結：【https://www.ywcnfztp.mom/#/】，我點入該電子郵件提供之連結後，依其要求輸入其所有之【車號、身分證字號、驗證碼等】，手機【有】接收到訊息認證碼9476，訊息認證碼輸入後，我的【信用卡】遭人【盜刷】新臺幣【33097】元後發覺遭詐騙，至所報案。"

checker = BlacklistManager()
result = checker.analyze(TEST_CASE)
print(result)