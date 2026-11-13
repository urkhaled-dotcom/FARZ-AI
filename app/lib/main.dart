import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'models/classification_result.dart';
import 'services/farz_inference_service.dart';
import 'theme/app_theme.dart';

void main() => runApp(const FarzApp());

class FarzApp extends StatefulWidget {
  const FarzApp({super.key});
  @override
  State<FarzApp> createState() => _FarzAppState();
}

class _FarzAppState extends State<FarzApp> {
  final _prefs = SharedPreferencesAsync();
  ThemeMode _theme = ThemeMode.system;
  Locale _locale = const Locale('ar');
  bool _loaded = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final theme = await _prefs.getString('themeMode');
    final language = await _prefs.getString('language');
    if (!mounted) return;
    setState(() {
      _theme = switch (theme) {
        'light' => ThemeMode.light,
        'dark' => ThemeMode.dark,
        _ => ThemeMode.system,
      };
      _locale = language == 'en' ? const Locale('en') : const Locale('ar');
      _loaded = true;
    });
  }

  Future<void> _setTheme(ThemeMode value) async {
    setState(() => _theme = value);
    await _prefs.setString('themeMode', switch (value) {
      ThemeMode.light => 'light',
      ThemeMode.dark => 'dark',
      ThemeMode.system => 'system',
    });
  }

  Future<void> _toggleLanguage() async {
    final value = _locale.languageCode == 'ar' ? const Locale('en') : const Locale('ar');
    setState(() => _locale = value);
    await _prefs.setString('language', value.languageCode);
  }

  @override
  Widget build(BuildContext context) {
    if (!_loaded) {
      return MaterialApp(
        theme: buildFarzLightTheme(),
        darkTheme: buildFarzDarkTheme(),
        home: const Scaffold(body: Center(child: CircularProgressIndicator())),
      );
    }
    final arabic = _locale.languageCode == 'ar';
    return MaterialApp(
      title: 'FARZ | فرز',
      debugShowCheckedModeBanner: false,
      theme: buildFarzLightTheme(),
      darkTheme: buildFarzDarkTheme(),
      themeMode: _theme,
      locale: _locale,
      supportedLocales: const [Locale('ar'), Locale('en')],
      builder: (context, child) => Directionality(
        textDirection: arabic ? TextDirection.rtl : TextDirection.ltr,
        child: child ?? const SizedBox.shrink(),
      ),
      home: FarzHomePage(
        isArabic: arabic,
        themeMode: _theme,
        onThemeChanged: _setTheme,
        onLanguageToggle: _toggleLanguage,
      ),
    );
  }
}

class FarzHomePage extends StatefulWidget {
  const FarzHomePage({
    required this.isArabic,
    required this.themeMode,
    required this.onThemeChanged,
    required this.onLanguageToggle,
    super.key,
  });

  final bool isArabic;
  final ThemeMode themeMode;
  final ValueChanged<ThemeMode> onThemeChanged;
  final VoidCallback onLanguageToggle;

  @override
  State<FarzHomePage> createState() => _FarzHomePageState();
}

class _FarzHomePageState extends State<FarzHomePage> {
  final _picker = ImagePicker();
  final _inference = FarzInferenceService();
  final _prefs = SharedPreferencesAsync();
  File? _image;
  ClassificationResult? _result;
  bool _modelReady = false;
  bool _checking = true;
  bool _busy = false;
  String? _error;
  List<Map<String, dynamic>> _history = [];

  bool get ar => widget.isArabic;
  String tr(String a, String e) => ar ? a : e;
  static const _keys = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal'];
  static const _arNames = ['زجاج', 'ورق', 'كرتون', 'بلاستيك', 'معادن'];
  static const _enNames = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal'];

  String className(String name) {
    final index = _keys.indexOf(name);
    if (index < 0) return name;
    return ar ? _arNames[index] : _enNames[index];
  }

  @override
  void initState() {
    super.initState();
    _checkModel();
    _loadHistory();
  }

  Future<void> _checkModel() async {
    final ready = await _inference.isModelReady();
    if (mounted) setState(() {
      _modelReady = ready;
      _checking = false;
    });
  }

  Future<void> _loadHistory() async {
    final values = await _prefs.getStringList('classificationHistory') ?? [];
    final entries = <Map<String, dynamic>>[];
    for (final value in values) {
      try {
        final decoded = jsonDecode(value);
        if (decoded is Map<String, dynamic>) entries.add(decoded);
      } catch (_) {
        // Ignore one malformed local entry.
      }
    }
    if (mounted) setState(() => _history = entries);
  }

  Future<void> _saveHistory(ClassificationResult result) async {
    final item = <String, dynamic>{
      'label': result.label,
      'score': result.score,
      'date': DateTime.now().toIso8601String(),
    };
    final next = [item, ..._history].take(20).toList();
    setState(() => _history = next);
    await _prefs.setStringList('classificationHistory', next.map(jsonEncode).toList());
  }

  Future<void> _pick(ImageSource source) async {
    try {
      final selected = await _picker.pickImage(
        source: source,
        imageQuality: 95,
        maxWidth: 1600,
        maxHeight: 1600,
      );
      if (selected == null || !mounted) return;
      setState(() {
        _image = File(selected.path);
        _result = null;
        _error = null;
      });
    } catch (error) {
      if (mounted) setState(() => _error = tr('تعذر فتح الصورة: $error', 'Could not open image: $error'));
    }
  }

  Future<void> _classify() async {
    final image = _image;
    if (image == null || _busy) return;
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
    });
    try {
      final value = await _inference.classify(image);
      if (!mounted) return;
      setState(() => _result = value);
      await _saveHistory(value);
    } catch (error) {
      if (mounted) setState(() => _error = _friendlyError(error));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _friendlyError(Object error) {
    final text = error.toString();
    if (text.contains('MODEL_NOT_FOUND') || text.contains('model file')) {
      return tr(
        'ملف النموذج غير مدمج داخل التطبيق بعد. أضف farz_model.pte إلى assets/models ثم أعد البناء.',
        'The model is not bundled yet. Add farz_model.pte to assets/models and rebuild the APK.',
      );
    }
    return tr('حدث خطأ أثناء التصنيف: $text', 'Classification failed: $text');
  }

  String _tip(String label) => switch (label) {
    'Glass' => tr('أفرغ العبوة واشطفها عند الحاجة، ثم استخدم مسار جمع الزجاج المحلي.', 'Empty and rinse if needed, then use the local glass collection stream.'),
    'Paper' => tr('حافظ على الورق جافاً ونظيفاً، وأزل بقايا الطعام قبل فرزه.', 'Keep paper dry and clean; remove food residue before recycling.'),
    'Cardboard' => tr('افرد الصندوق وأزل الشريط والمواد غير الورقية قدر الإمكان.', 'Flatten the box and remove tape or non-paper parts where practical.'),
    'Plastic' => tr('أفرغ العبوة واشطفها عند الحاجة، واتبع قواعد جمع البلاستيك المحلية.', 'Empty and rinse if needed; follow local plastic collection rules.'),
    'Metal' => tr('أفرغ العلبة واشطفها عند الحاجة، وتحقق من تعليمات جمع المعادن في منطقتك.', 'Empty and rinse if needed; check local metal collection rules.'),
    _ => tr('تحقق من تعليمات الفرز المحلية.', 'Check local sorting guidance.'),
  };

  Future<void> _showHistory() async {
    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      builder: (sheetContext) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(tr('سجل التصنيفات', 'Classification history'),
                style: Theme.of(sheetContext).textTheme.titleLarge),
              const SizedBox(height: 12),
              if (_history.isEmpty)
                Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text(tr('لا توجد نتائج محفوظة بعد.', 'No saved results yet.'),
                    textAlign: TextAlign.center),
                )
              else
                Flexible(
                  child: ListView.separated(
                    shrinkWrap: true,
                    itemCount: _history.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (_, index) {
                      final item = _history[index];
                      final date = DateTime.tryParse(item['date'] as String? ?? '')?.toLocal();
                      final score = (item['score'] as num?)?.toDouble() ?? 0;
                      return ListTile(
                        leading: const Icon(Icons.recycling),
                        title: Text(className(item['label'] as String? ?? '')),
                        subtitle: Text(date == null ? '' : date.toString().substring(0, 16)),
                        trailing: Text('${(score * 100).toStringAsFixed(1)}%'),
                      );
                    },
                  ),
                ),
              if (_history.isNotEmpty)
                TextButton.icon(
                  onPressed: () async {
                    await _prefs.remove('classificationHistory');
                    if (mounted) setState(() => _history = []);
                    if (sheetContext.mounted) Navigator.pop(sheetContext);
                  },
                  icon: const Icon(Icons.delete_outline),
                  label: Text(tr('مسح السجل', 'Clear history')),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.recycling_rounded, size: 29, color: colors.primary),
          const SizedBox(width: 8),
          Text('FARZ', style: TextStyle(fontWeight: FontWeight.w900,
            letterSpacing: 1.2, color: colors.primary)),
        ]),
        actions: [
          IconButton(tooltip: tr('السجل', 'History'), onPressed: _showHistory,
            icon: const Icon(Icons.history)),
          PopupMenuButton<ThemeMode>(
            tooltip: tr('المظهر', 'Appearance'),
            icon: Icon(widget.themeMode == ThemeMode.dark ? Icons.dark_mode : Icons.light_mode_outlined),
            onSelected: widget.onThemeChanged,
            itemBuilder: (_) => [
              PopupMenuItem(value: ThemeMode.system, child: Text(tr('حسب الجهاز', 'System'))),
              PopupMenuItem(value: ThemeMode.light, child: Text(tr('نهاري', 'Light'))),
              PopupMenuItem(value: ThemeMode.dark, child: Text(tr('ليلي', 'Dark'))),
            ],
          ),
          TextButton(onPressed: widget.onLanguageToggle, child: Text(ar ? 'EN' : 'ع')),
          const SizedBox(width: 6),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(18, 12, 18, 28),
          children: [
            Text(tr('مرحباً بك في فَرز', 'Welcome to FARZ'),
              style: theme.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800)),
            const SizedBox(height: 6),
            Text(
              tr('صوّر قطعة نفايات واحدة، وسنساعدك في التعرّف على مادتها.',
                'Take a photo of one waste item and identify its material.'),
              style: theme.textTheme.bodyLarge?.copyWith(color: colors.onSurfaceVariant, height: 1.45),
            ),
            const SizedBox(height: 18),
            _modelStatus(theme),
            const SizedBox(height: 16),
            _imagePreview(theme),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: OutlinedButton.icon(
                onPressed: _busy ? null : () => _pick(ImageSource.camera),
                icon: const Icon(Icons.camera_alt_outlined),
                label: Text(tr('الكاميرا', 'Camera')),
                style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(52)),
              )),
              const SizedBox(width: 10),
              Expanded(child: OutlinedButton.icon(
                onPressed: _busy ? null : () => _pick(ImageSource.gallery),
                icon: const Icon(Icons.photo_library_outlined),
                label: Text(tr('المعرض', 'Gallery')),
                style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(52)),
              )),
            ]),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: (_image == null || _busy || !_modelReady) ? null : _classify,
              icon: _busy
                ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(Icons.auto_awesome),
              label: Text(_busy ? tr('جارٍ التصنيف...', 'Classifying...') : tr('تعرّف على المادة', 'Identify material')),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              _message(theme, _error!),
            ],
            if (_result != null) ...[
              const SizedBox(height: 18),
              _resultCard(theme, _result!),
            ],
            const SizedBox(height: 22),
            Text(tr('الفئات التي يتعرف عليها FARZ', 'FARZ material classes'),
              style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
            const SizedBox(height: 10),
            _classGrid(theme),
            const SizedBox(height: 14),
            Center(child: Text(
              tr('التصنيف يعمل على هاتفك دون إرسال الصور إلى الإنترنت.',
                'Classification runs on your device. Photos are not sent online.'),
              textAlign: TextAlign.center,
              style: theme.textTheme.bodySmall?.copyWith(color: colors.primary, fontWeight: FontWeight.w600),
            )),
          ],
        ),
      ),
    );
  }

  Widget _modelStatus(ThemeData theme) {
    final ready = _modelReady;
    final color = ready ? theme.colorScheme.primary : theme.colorScheme.tertiary;
    final message = _checking
      ? tr('جارٍ فحص النموذج المحلي...', 'Checking on-device model...')
      : ready
        ? tr('نموذج الذكاء الاصطناعي جاهز على الهاتف', 'On-device AI model is ready')
        : tr('ملف النموذج لم يُدمج بعد', 'Model file has not been bundled yet');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(color: color.withValues(alpha: 0.09),
        borderRadius: BorderRadius.circular(16)),
      child: Row(children: [
        Icon(ready ? Icons.verified_rounded : Icons.info_outline, color: color),
        const SizedBox(width: 10),
        Expanded(child: Text(message, style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600))),
        if (!_checking && !ready)
          IconButton(tooltip: tr('إعادة الفحص', 'Retry'), onPressed: () async {
            setState(() => _checking = true);
            await _checkModel();
          }, icon: const Icon(Icons.refresh)),
      ]),
    );
  }

  Widget _imagePreview(ThemeData theme) => Card(
    clipBehavior: Clip.antiAlias,
    child: Container(
      height: 232,
      width: double.infinity,
      color: theme.colorScheme.primary.withValues(alpha: 0.055),
      child: _image == null
        ? Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(Icons.add_a_photo_outlined, size: 54, color: theme.colorScheme.primary),
            const SizedBox(height: 10),
            Text(tr('اختر صورة لبدء التحليل', 'Choose an image to begin'), style: theme.textTheme.titleSmall),
            const SizedBox(height: 5),
            Text(tr('صورة واحدة وقطعة واحدة في كل مرة', 'One item per image'),
              style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ])
        : Stack(fit: StackFit.expand, children: [
            Image.file(_image!, fit: BoxFit.contain),
            Positioned(top: 8, right: 8, child: IconButton.filledTonal(
              onPressed: () => setState(() { _image = null; _result = null; _error = null; }),
              icon: const Icon(Icons.close),
            )),
          ]),
    ),
  );

  Widget _resultCard(ThemeData theme, ClassificationResult result) {
    final color = theme.colorScheme;
    return Card(child: Padding(
      padding: const EdgeInsets.all(18),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          Icon(Icons.check_circle, color: color.primary),
          const SizedBox(width: 8),
          Text(tr('نتيجة التصنيف', 'Classification result'),
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
        ]),
        const SizedBox(height: 14),
        Text(className(result.label),
          style: theme.textTheme.headlineLarge?.copyWith(color: color.primary, fontWeight: FontWeight.w900)),
        const SizedBox(height: 4),
        Text('${tr('درجة النموذج', 'Model score')}: ${(result.score * 100).toStringAsFixed(1)}%'),
        const SizedBox(height: 12),
        ...result.scores.map((score) => _scoreRow(theme, score)),
        const SizedBox(height: 12),
        Text(tr('إرشادات الفرز', 'Sorting guidance'),
          style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(height: 5),
        Text(_tip(result.label), style: theme.textTheme.bodyMedium?.copyWith(height: 1.5)),
        const SizedBox(height: 10),
        Text(
          tr('درجات النموذج ليست احتمالات معايرة. قد تتأثر النتيجة بالإضاءة والخلفية وجودة الصورة.',
            'Model scores are not calibrated probabilities. Lighting, background and image quality can affect the result.'),
          style: theme.textTheme.bodySmall?.copyWith(color: color.onSurfaceVariant),
        ),
        Align(alignment: AlignmentDirectional.centerEnd,
          child: Text('${result.latencyMs.toStringAsFixed(0)} ms',
            style: theme.textTheme.labelSmall?.copyWith(color: color.onSurfaceVariant))),
      ]),
    ));
  }

  Widget _scoreRow(ThemeData theme, ClassScore score) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Row(children: [
      SizedBox(width: 88, child: Text(className(score.label), style: theme.textTheme.bodySmall)),
      Expanded(child: ClipRRect(
        borderRadius: BorderRadius.circular(99),
        child: LinearProgressIndicator(
          value: score.score.clamp(0, 1),
          minHeight: 8,
          backgroundColor: theme.colorScheme.surfaceContainerHighest,
        ),
      )),
      const SizedBox(width: 8),
      SizedBox(width: 48, child: Text('${(score.score * 100).toStringAsFixed(1)}%',
        textAlign: TextAlign.end, style: theme.textTheme.labelSmall)),
    ]),
  );

  Widget _classGrid(ThemeData theme) {
    const icons = [
      Icons.wine_bar_outlined, Icons.description_outlined,
      Icons.inventory_2_outlined, Icons.local_drink_outlined, Icons.recycling,
    ];
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: List.generate(_keys.length, (index) => SizedBox(
        width: (MediaQuery.sizeOf(context).width - 52) / 2,
        child: Card(child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          child: Row(children: [
            Icon(icons[index], color: theme.colorScheme.primary),
            const SizedBox(width: 8),
            Flexible(child: Text(ar ? _arNames[index] : _enNames[index],
              style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600))),
          ]),
        )),
      )),
    );
  }

  Widget _message(ThemeData theme, String message) {
    final color = theme.colorScheme.error;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(color: color.withValues(alpha: 0.08), borderRadius: BorderRadius.circular(14)),
      child: Text(message, style: theme.textTheme.bodyMedium?.copyWith(color: color)),
    );
  }
}
