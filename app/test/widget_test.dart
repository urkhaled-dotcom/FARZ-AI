import 'package:flutter_test/flutter_test.dart';
import 'package:farz_app/main.dart';

void main() {
  testWidgets('FARZ starts in Arabic and shows all five classes', (tester) async {
    await tester.pumpWidget(const FarzApp());

    expect(find.text('FARZ'), findsOneWidget);
    expect(find.text('مرحباً بك في فَرز'), findsOneWidget);
    expect(find.text('زجاج'), findsOneWidget);
    expect(find.text('ورق'), findsOneWidget);
    expect(find.text('كرتون'), findsOneWidget);
    expect(find.text('بلاستيك'), findsOneWidget);
    expect(find.text('معادن'), findsOneWidget);
  });

  testWidgets('language switch changes labels to English', (tester) async {
    await tester.pumpWidget(const FarzApp());

    await tester.tap(find.text('EN'));
    await tester.pumpAndSettle();

    expect(find.text('Welcome to FARZ'), findsOneWidget);
    expect(find.text('Glass'), findsOneWidget);
    expect(find.text('Paper'), findsOneWidget);
    expect(find.text('Cardboard'), findsOneWidget);
    expect(find.text('Plastic'), findsOneWidget);
    expect(find.text('Metal'), findsOneWidget);
  });

  testWidgets('theme control switches from system to dark mode', (tester) async {
    await tester.pumpWidget(const FarzApp());

    await tester.tap(find.byTooltip('تغيير المظهر'));
    await tester.pumpAndSettle();

    expect(find.byIcon(Icons.dark_mode), findsOneWidget);
  });
}
