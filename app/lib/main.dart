import 'package:flutter/material.dart';

import 'theme/app_theme.dart';

void main() => runApp(const FarzApp());

class FarzApp extends StatefulWidget {
  const FarzApp({super.key});

  @override
  State<FarzApp> createState() => _FarzAppState();
}

class _FarzAppState extends State<FarzApp> {
  ThemeMode _themeMode = ThemeMode.system;
  Locale _locale = const Locale('ar');

  void _toggleTheme() {
    setState(() {
      _themeMode = switch (_themeMode) {
        ThemeMode.system => ThemeMode.dark,
        ThemeMode.dark => ThemeMode.light,
        ThemeMode.light => ThemeMode.system,
      };
    });
  }

  void _toggleLanguage() {
    setState(() {
      _locale = _locale.languageCode == 'ar'
          ? const Locale('en')
          : const Locale('ar');
    });
  }

  @override
  Widget build(BuildContext context) {
    final isArabic = _locale.languageCode == 'ar';
    return MaterialApp(
      title: 'FARZ | فرز',
      debugShowCheckedModeBanner: false,
      theme: buildFarzLightTheme(),
      darkTheme: buildFarzDarkTheme(),
      themeMode: _themeMode,
      locale: _locale,
      builder: (context, child) => Directionality(
        textDirection: isArabic ? TextDirection.rtl : TextDirection.ltr,
        child: child ?? const SizedBox.shrink(),
      ),
      home: FarzHomePage(
        isArabic: isArabic,
        themeMode: _themeMode,
        onToggleTheme: _toggleTheme,
        onToggleLanguage: _toggleLanguage,
      ),
    );
  }
}

class FarzHomePage extends StatelessWidget {
  const FarzHomePage({
    required this.isArabic,
    required this.themeMode,
    required this.onToggleTheme,
    required this.onToggleLanguage,
    super.key,
  });

  final bool isArabic;
  final ThemeMode themeMode;
  final VoidCallback onToggleTheme;
  final VoidCallback onToggleLanguage;

  String tr(String ar, String en) => isArabic ? ar : en;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final textColor = theme.colorScheme.onSurface;
    final classes = isArabic
        ? const ['زجاج', 'ورق', 'كرتون', 'بلاستيك', 'معادن']
        : const ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal'];
    final icons = const [
      Icons.wine_bar_outlined,
      Icons.description_outlined,
      Icons.inventory_2_outlined,
      Icons.local_drink_outlined,
      Icons.recycling,
    ];

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.recycling, size: 32, color: theme.colorScheme.primary),
            const SizedBox(width: 8),
            Text(
              'FARZ',
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: theme.colorScheme.primary,
                letterSpacing: 1.3,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: tr('تغيير المظهر', 'Change appearance'),
            onPressed: onToggleTheme,
            icon: Icon(
              themeMode == ThemeMode.dark
                  ? Icons.dark_mode
                  : Icons.light_mode_outlined,
            ),
          ),
          TextButton(
            onPressed: onToggleLanguage,
            child: Text(isArabic ? 'EN' : 'ع'),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 18, 20, 28),
          children: [
            Text(
              tr('مرحباً بك في فَرز', 'Welcome to FARZ'),
              style: theme.textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.w800,
                color: textColor,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              tr(
                'صوّر النفاية، تعرّف على مادتها، واختر طريقة فرزها.',
                'Scan a waste item, identify its material, and learn how to sort it.',
              ),
              style: theme.textTheme.bodyLarge?.copyWith(
                color: textColor.withValues(alpha: 0.72),
                height: 1.5,
              ),
            ),
            const SizedBox(height: 26),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Container(
                      height: 150,
                      decoration: BoxDecoration(
                        color: theme.colorScheme.primary.withValues(alpha: 0.09),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Icon(
                        Icons.photo_camera_outlined,
                        size: 66,
                        color: theme.colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: 18),
                    FilledButton.icon(
                      onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(
                            tr(
                              'سيتم ربط الكاميرا في المرحلة التالية.',
                              'Camera integration is coming in the next step.',
                            ),
                          ),
                        ),
                      ),
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: Text(tr('ابدأ التصوير', 'Start scanning')),
                    ),
                    const SizedBox(height: 8),
                    TextButton.icon(
                      onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(
                            tr(
                              'سيتم إضافة اختيار الصور من المعرض لاحقاً.',
                              'Gallery selection will be added next.',
                            ),
                          ),
                        ),
                      ),
                      icon: const Icon(Icons.photo_library_outlined),
                      label: Text(tr('اختيار صورة من المعرض', 'Choose from gallery')),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 26),
            Text(
              tr('الفئات الخمس', 'Five material classes'),
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 12),
            ...List.generate(
              classes.length,
              (index) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Card(
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor:
                          theme.colorScheme.primary.withValues(alpha: 0.10),
                      child: Icon(icons[index], color: theme.colorScheme.primary),
                    ),
                    title: Text(
                      classes[index],
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    trailing: Icon(
                      isArabic ? Icons.chevron_left : Icons.chevron_right,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Center(
              child: Text(
                tr(
                  'يعمل FARZ دون اتصال بالإنترنت',
                  'FARZ works offline',
                ),
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.primary,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
