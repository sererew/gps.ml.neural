package uo.ml.neural.tracks.core.exception;

/**
 * Unchecked exception for CLI command errors.
 */
@SuppressWarnings("serial")
public class CommandException extends RuntimeException {
    public CommandException(String message) {
        super(message);
    }
    public CommandException(String message, Throwable cause) {
        super(message, cause);
    }
}