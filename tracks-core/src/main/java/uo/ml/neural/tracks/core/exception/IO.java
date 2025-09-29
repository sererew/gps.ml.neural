package uo.ml.neural.tracks.core.exception;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Utility class for handling IO operations with unchecked exceptions.
 */
public final class IO {
	public static <T> T get(IOSupplier<T> s) {
		try {
			return s.get();
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	@FunctionalInterface
	public interface IOSupplier<T> {
		T get() throws IOException;
	}

	public static <T> void shallow(IOShallow<T> s) {
		try {
			s.run();
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	@FunctionalInterface
	public interface IOShallow<T> {
		void run() throws IOException;
	}
}