import cv2
import numpy as np
import os


class BogoQualityChecker:
    def __init__(self, path):
        """
        Inicjalizuje obiekt ImageQualityChecker dla podanego pliku obrazu.

        Sprawdza, czy plik istnieje, ma prawidłowe rozszerzenie oraz czy może być poprawnie wczytany.
        """
        self.image_path = path  # Ścieżka do pliku obrazu

        # Sprawdzenie, czy plik istnieje
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Plik {path} nie istnieje.')

        # Sprawdzenie rozszerzenia pliku w stosunku do dozwolonych
        allowed_extensions = os.environ.get('ALLOWED_EXTENSIONS', [".jpg", ".jpeg", ".png"])
        if os.path.splitext(path)[1].lower() not in allowed_extensions:
            raise TypeError(f'Ten rodzaj pliku nie jest obsługiwany.')

        # Wczytanie obrazu w kolorze (ważne dla testów saturacji i naświetlenia)
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"Nie można wczytać pliku {path} jako obrazu.")

    def check_resolution(self, min_width=600, min_height=600):
        """Sprawdza, czy obraz spełnia minimalne wymagania dotyczące rozdzielczości."""
        # Pobiera wysokość i szerokość obrazu
        height, width = self.image.shape[:2]
        # Sprawdza, czy szerokość i wysokość są co najmniej równe minimalnym wartościom
        return width >= min_width and height >= min_height

    def estimate_blur(self):
        """Szacuje poziom rozmycia obrazu za pomocą wariancji Laplacjanu."""
        # Konwertuje obraz na skalę szarości
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Oblicza Laplacjan obrazu (wykrywanie krawędzi)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Oblicza wariancję Laplacjanu (im mniejsza wartość, tym bardziej rozmyty obraz)
        laplacian_var = laplacian.var()
        return laplacian_var

    def check_blurriness(self, blur_threshold=50):
        """Sprawdza, czy obraz jest wystarczająco ostry."""
        # Szacuje rozmycie obrazu
        blur_value = self.estimate_blur()
        # Porównuje wartość rozmycia z progiem
        return blur_value >= blur_threshold

    def check_brightness(self, lower_threshold=30, upper_threshold=220):
        """Sprawdza, czy jasność obrazu mieści się w akceptowalnym zakresie."""
        # Konwertuje obraz na skalę szarości
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Oblicza średnią jasność
        mean_brightness = np.mean(gray)
        # Sprawdza, czy średnia jasność mieści się pomiędzy dolnym a górnym progiem
        return lower_threshold < mean_brightness < upper_threshold

    def check_saturation(self, threshold=20):
        """Sprawdza, czy obraz ma wystarczającą saturację kolorów."""
        # Konwertuje obraz do przestrzeni barw HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Pobiera kanał saturacji
        saturation = hsv[:, :, 1]
        # Oblicza średnią saturację
        mean_saturation = np.mean(saturation)
        # Sprawdza, czy średnia saturacja jest większa niż próg
        return mean_saturation > threshold

    def estimate_noise(self):
        """Szacuje ilość szumu w obrazie."""
        # Konwertuje obraz na skalę szarości
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Oblicza różnicę między obrazem a rozmytą wersją obrazu (filtr Gaussa)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = gray - blur
        # Oblicza odchylenie standardowe szumu
        noise_std = np.std(noise)
        return noise_std

    def check_noise(self, noise_threshold=10):
        """Sprawdza, czy poziom szumu w obrazie jest akceptowalny."""
        # Szacuje poziom szumu
        noise_value = self.estimate_noise()
        # Sprawdza, czy poziom szumu jest poniżej progu
        return noise_value <= noise_threshold

    def check_exposure(self, lower_percentile=5, upper_percentile=95):
        """Sprawdza, czy obraz jest prawidłowo naświetlony."""
        # Konwertuje obraz na skalę szarości
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Spłaszcza tablicę do jednego wymiaru
        pixels = gray.flatten()
        # Oblicza percentyle
        lower = np.percentile(pixels, lower_percentile)
        upper = np.percentile(pixels, upper_percentile)
        # Sprawdza, czy zakres tonalny jest wystarczający
        return (upper - lower) > 50  # Próg można dostosować w zależności od wymagań

    def classify_quality(self):
        """Klasyfikuje jakość obrazu na podstawie różnych metryk i zwraca wynik jakości."""
        tests = []  # Lista przechowująca wyniki poszczególnych testów

        # Sprawdza rozdzielczość, ale nie wlicza do wyniku jakości
        resolution_passed = self.check_resolution()

        # Sprawdza ostrość (rozmycie)
        blurriness_passed = self.check_blurriness()
        tests.append(('blurriness', blurriness_passed))

        # Sprawdza jasność
        brightness_passed = self.check_brightness()
        tests.append(('brightness', brightness_passed))

        # Sprawdza saturację kolorów
        saturation_passed = self.check_saturation()
        tests.append(('saturation', saturation_passed))

        # Sprawdza poziom szumu
        noise_passed = self.check_noise()
        tests.append(('noise', noise_passed))

        # Sprawdza naświetlenie
        exposure_passed = self.check_exposure()
        tests.append(('exposure', exposure_passed))

        # Oblicza liczbę zaliczonych testów i całkowitą liczbę testów
        tests_passed = sum(1 for test_name, passed in tests if passed)
        total_tests = len(tests)

        # Oblicza wynik jakości (proporcja zaliczonych testów)
        quality_score = tests_passed / total_tests

        # Przygotowuje szczegółowe wyniki
        results = {
            'image_path': self.image_path,
            'quality_score': quality_score,
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'resolution_passed': resolution_passed,
        }

        return results