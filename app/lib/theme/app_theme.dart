import 'package:flutter/material.dart';

abstract final class FarzColors {
  static const forest = Color(0xFF176B50);
  static const mint = Color(0xFFBCE8D4);
  static const ecoWhite = Color(0xFFF5F7F2);
  static const charcoal = Color(0xFF24332D);
  static const amber = Color(0xFFE6A23C);
  static const softSage = Color(0xFFE6EFE9);
  static const darkBackground = Color(0xFF17251F);
  static const darkPrimary = Color(0xFF8DD4B2);
}

ThemeData buildFarzLightTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: FarzColors.forest,
    brightness: Brightness.light,
  ).copyWith(
    primary: FarzColors.forest,
    onPrimary: Colors.white,
    secondary: FarzColors.mint,
    surface: Colors.white,
  );

  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: FarzColors.ecoWhite,
    appBarTheme: const AppBarTheme(
      backgroundColor: FarzColors.ecoWhite,
      foregroundColor: FarzColors.charcoal,
      centerTitle: false,
      elevation: 0,
    ),
    cardTheme: CardThemeData(
      color: Colors.white,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(24),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: FarzColors.forest,
        foregroundColor: Colors.white,
        minimumSize: const Size.fromHeight(54),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
        ),
      ),
    ),
  );
}

ThemeData buildFarzDarkTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: FarzColors.darkPrimary,
    brightness: Brightness.dark,
  ).copyWith(
    primary: FarzColors.darkPrimary,
    onPrimary: FarzColors.darkBackground,
    secondary: FarzColors.mint,
    surface: FarzColors.charcoal,
  );

  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: FarzColors.darkBackground,
    appBarTheme: const AppBarTheme(
      backgroundColor: FarzColors.darkBackground,
      foregroundColor: Color(0xFFF2F5F1),
      centerTitle: false,
      elevation: 0,
    ),
    cardTheme: CardThemeData(
      color: FarzColors.charcoal,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(24),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: const Color(0xFF267957),
        foregroundColor: Colors.white,
        minimumSize: const Size.fromHeight(54),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
        ),
      ),
    ),
  );
}
