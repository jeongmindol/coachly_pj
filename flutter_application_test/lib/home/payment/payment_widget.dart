import 'package:bootpay/bootpay.dart';
import 'package:bootpay/model/extra.dart';
import 'package:bootpay/model/item.dart';
import 'package:bootpay/model/payload.dart';
import 'package:bootpay/model/user.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class PaymentWidget extends StatelessWidget {
  const PaymentWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("운동 장소 선택"),
        backgroundColor: const Color.fromARGB(255, 255, 111, 97),
        actions: [
          TextButton(
            onPressed: () {}, // 상태 초기화 및 홈 페이지로 이동
            child: const Text(
              "다음에 이용하기",
              style: TextStyle(color: Colors.black),
            ),
          ),
        ],
      ),
      body: TextButton(
          onPressed: () => bootpayTest(context),
          child: const Text('정기결제 테스트(인증)', style: TextStyle(fontSize: 16.0))),
    );
  }
}

void bootpayTest(BuildContext context) {
  Payload payload = getPayload();
  if (kIsWeb) {
    payload.extra?.openType = "iframe";
  }

  Bootpay().requestSubscription(
    context: context,
    payload: payload,
    showCloseButton: false,
    // closeButton: Icon(Icons.close, size: 35.0, color: Colors.black54),
    onCancel: (String data) {
      print('------- onCancel: $data');
    },
    onError: (String data) {
      print('------- onCancel: $data');
    },
    onClose: () {
      print('------- onClose');
      Bootpay().dismiss(context); //명시적으로 부트페이 뷰 종료 호출
      //TODO - 원하시는 라우터로 페이지 이동
    },
    onIssued: (String data) {
      print('------- onIssued: $data');
    },
    onConfirm: (String data) {
      /**
            1. 바로 승인하고자 할 때
            return true;
         **/
      /***
            2. 비동기 승인 하고자 할 때
            checkQtyFromServer(data);
            return false;
         ***/
      /***
            3. 서버승인을 하고자 하실 때 (클라이언트 승인 X)
            return false; 후에 서버에서 결제승인 수행
         */
      // checkQtyFromServer(data);
      return true;
    },
    onDone: (String data) {
      print('------- onDone: $data');
    },
  );
}

Payload getPayload() {
  Payload payload = Payload();
  Item item1 = Item();
  item1.name = "미키 '마우스"; // 주문정보에 담길 상품명
  item1.qty = 1; // 해당 상품의 주문 수량
  item1.id = "ITEM_CODE_MOUSE"; // 해당 상품의 고유 키
  item1.price = 500; // 상품의 가격

  Item item2 = Item();
  item2.name = "키보드"; // 주문정보에 담길 상품명
  item2.qty = 1; // 해당 상품의 주문 수량
  item2.id = "ITEM_CODE_KEYBOARD"; // 해당 상품의 고유 키
  item2.price = 500; // 상품의 가격
  List<Item> itemList = [item1, item2];

  payload.androidApplicationId =
      '67626cfbcc5274a3ac3fc91a'; // android application id
  payload.iosApplicationId = '67626cfbcc5274a3ac3fc91b'; // ios application id

  payload.pg = '나이스페이';
  payload.method = '카드자동';
  // payload.methods = ['card', 'phone', 'vbank', 'bank', 'kakao'];
  payload.orderName = "테스트 상품"; //결제할 상품명
  payload.price = 1000.0; //정기결제시 0 혹은 주석

  // payload.orderId = DateTime.now().millisecondsSinceEpoch.toString(); //주문번호, 개발사에서 고유값으로 지정해야함
  payload.subscriptionId = DateTime.now()
      .millisecondsSinceEpoch
      .toString(); //주문번호, 개발사에서 고유값으로 지정해야함

  payload.metadata = {
    "callbackParam1": "value12",
    "callbackParam2": "value34",
    "callbackParam3": "value56",
    "callbackParam4": "value78",
  }; // 전달할 파라미터, 결제 후 되돌려 주는 값
  payload.items = itemList; // 상품정보 배열

  User user = User(); // 구매자 정보
  user.username = "사용자 이름";
  user.email = "user1234@gmail.com";
  user.area = "서울";
  user.phone = "010-4033-4678";
  user.addr = '서울시 동작구 상도로 222';

  Extra extra = Extra(); // 결제 옵션
  extra.appScheme = 'bootpayFlutterExample';
  extra.cardQuota = '3';
  // extra.openType = 'popup';

  // extra.carrier = "SKT,KT,LGT"; //본인인증 시 고정할 통신사명
  // extra.ageLimit = 20; // 본인인증시 제한할 최소 나이 ex) 20 -> 20살 이상만 인증이 가능

  payload.user = user;
  payload.extra = extra;
  return payload;
}
